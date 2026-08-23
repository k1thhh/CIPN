//model.h must be a separate file in same sketch folder.

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <Adafruit_DRV2605.h>

#include <tflm_esp32.h>
#include <eloquent_tinyml.h>

#include "model.h" // keep model binary in separate file

// -------------------- OLED --------------------
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);
const uint8_t OLED_ADDR = 0x3C;

// -------------------- DRV2605 (haptic) --------------------
Adafruit_DRV2605 drv;
bool drvInitialized = false;
const int DRV_EN_PIN = 32; // change if your breakout uses different enable polarity

// -------------------- Buttons --------------------
// Option buttons (1..4)
const int BTN1_PIN = 13;
const int BTN2_PIN = 25;
const int BTN3_PIN = 14;
const int BTN4_PIN = 27;
// Reset / response button
const int BTN_RST_PIN = 26;

const int NUM_BUTTONS = 5;
const int btnPins[NUM_BUTTONS] = { BTN1_PIN, BTN2_PIN, BTN3_PIN, BTN4_PIN, BTN_RST_PIN };

// Debounce
const unsigned long DEBOUNCE_MS = 50;
bool lastStableState[NUM_BUTTONS];
bool lastReading[NUM_BUTTONS];
unsigned long lastDebounceTimeArr[NUM_BUTTONS];
unsigned long pressTimeArr[NUM_BUTTONS];
bool reportedPress[NUM_BUTTONS];

// -------------------- TF model --------------------
#define ARENA_SIZE (32 * 1024)
#define MAX_TF_OPS 20
Eloquent::TF::Sequential<MAX_TF_OPS, ARENA_SIZE> tf;

// Preproc arrays (your JSON values)
const int N_FEATURES = 5;
const float IMP_MEDIAN[N_FEATURES] = {4.945f, 30.29f, 22.2f, 20.8f, 11.1f};
const float SCALER_MEAN[N_FEATURES] = {
  5.225890909090909f,
  30.24419545454546f,
  23.29072727272727f,
  20.745727272727272f,
  14.038181818181819f
};
const float SCALER_SCALE[N_FEATURES] = {
  2.4924283388821453f,
  1.4113371763805715f,
  10.069571418982212f,
  10.306434879511917f,
  15.321920315311665f
};

// -------------------- Questions --------------------
enum Group { G_SENSORY, G_MOTOR, G_AUTONOMIC };

struct Q {
  const char *text;
  Group group;
};

Q questions[] = {
  {"Tingling fingers or hands?", G_SENSORY},
  {"Tingling toes or feet?",    G_SENSORY},
  {"Numbness in fingers or hands?", G_SENSORY},
  {"Numbness in toes or feet?", G_SENSORY},
  {"Shooting/burning pain in hands?", G_SENSORY},
  {"Shooting/burning pain in feet?", G_SENSORY},
  {"Cramps in hands?", G_MOTOR},
  {"Cramps in feet?", G_MOTOR},
  {"Difficulty climbing stairs due to weakness?", G_MOTOR},
  {"Dizzy when standing up?", G_AUTONOMIC},
  {"Blurred vision?", G_AUTONOMIC},
  {"Difficulty hearing?", G_AUTONOMIC},
};
const int NUM_QUESTIONS = sizeof(questions) / sizeof(questions[0]);
uint8_t answers[NUM_QUESTIONS];
int current_question = 0;
bool finished = false;

// -------------------- UI helpers --------------------
void showTextPage(const char *line1, const char *line2 = nullptr, const char *line3 = nullptr, const char *line4 = nullptr) {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  int y = 0;
  if (line1) { display.setCursor(0, y); display.println(line1); y += 10; }
  if (line2) { display.setCursor(0, y); display.println(line2); y += 10; }
  if (line3) { display.setCursor(0, y); display.println(line3); y += 10; }
  if (line4) { display.setCursor(0, y); display.println(line4); y += 10; }
  display.display();
}

void showQuestion(int idx) {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0,0);
  String t = String(questions[idx].text);
  int wrap = 24;
  if (t.length() <= wrap) {
    display.println(t);
  } else {
    display.println(t.substring(0, wrap));
    display.println(t.substring(wrap));
  }
  display.println();
  display.println("Press button 1..4");
  display.println("1:Not at all  4:Very much");
  display.display();
}

// -------------------- Buttons --------------------
void initButtons() {
  for (int i = 0; i < NUM_BUTTONS; ++i) {
    pinMode(btnPins[i], INPUT_PULLUP);
    lastStableState[i] = digitalRead(btnPins[i]);
    lastReading[i] = lastStableState[i];
    lastDebounceTimeArr[i] = 0;
    pressTimeArr[i] = 0;
    reportedPress[i] = false;
  }
}

void updateButtons() {
  unsigned long now = millis();
  for (int i = 0; i < NUM_BUTTONS; ++i) {
    bool reading = digitalRead(btnPins[i]);
    if (reading != lastReading[i]) {
      lastDebounceTimeArr[i] = now;
      lastReading[i] = reading;
    } else {
      if ((now - lastDebounceTimeArr[i]) >= DEBOUNCE_MS) {
        if (reading != lastStableState[i]) {
          lastStableState[i] = reading;
          if (reading == LOW) {
            pressTimeArr[i] = now;
            reportedPress[i] = true;
            Serial.printf("btn[%d] PRESSED\n", i);
          } else {
            // silent on release
          }
        }
      }
    }
  }
}

int consumePressEvent() {
  for (int i = 0; i < NUM_BUTTONS; ++i) {
    if (reportedPress[i]) {
      reportedPress[i] = false;
      return i;
    }
  }
  return -1;
}

void waitForChoiceAndStore(int qidx) {
  showQuestion(qidx);
  Serial.printf("Waiting answer for Q%d: %s\n", qidx+1, questions[qidx].text);
  while (true) {
    updateButtons();
    int ev = consumePressEvent();
    if (ev >= 0) {
      if (ev <= 3) {
        answers[qidx] = (uint8_t)(ev + 1);
        Serial.printf("Recorded answer Q%d = %d\n", qidx+1, answers[qidx]);
        char buf[32];
        snprintf(buf, sizeof(buf), "Answer saved: %d", answers[qidx]);
        showTextPage(buf);
        delay(350);
        return;
      } else if (ev == 4) {
        // reset pressed: restart entire questionnaire
        showTextPage("Restarting questionnaire...");
        memset(answers, 0, sizeof(answers));
        current_question = 0;
        delay(400);
        showQuestion(current_question);
      }
    }
    delay(10);
  }
}

// -------------------- Sum & norm --------------------
float sum_group_answers(Group g) {
  float s = 0;
  for (int i = 0; i < NUM_QUESTIONS; ++i) {
    if (questions[i].group == g) s += (float)answers[i];
  }
  return s;
}

float normalize_feature(int idx, float raw) {
  if (!isfinite(raw)) raw = IMP_MEDIAN[idx];
  return (raw - SCALER_MEAN[idx]) / SCALER_SCALE[idx];
}

// -------------------- Inference helpers --------------------
void prepare_and_run_model(); // forward decl

bool run_inference_and_show(float sample_norm[]) {
  Eloquent::Error::Exception &pred_exc = tf.predict(sample_norm);
  if (!pred_exc.isOk()) {
    String s = pred_exc.toString();
    char buf[64]; s.toCharArray(buf, sizeof(buf));
    showTextPage("Inference failed:", buf);
    Serial.print("Inference failed: ");
    Serial.println(s);
    return false;
  }
  float prob = tf.output(0);
  char buf1[32], buf2[32];
  snprintf(buf1, sizeof(buf1), "Prob = %.6f", prob);
  snprintf(buf2, sizeof(buf2), "Feat0(VPT)=%.2f", sample_norm[0]);
  showTextPage("Result", buf1, buf2);
  Serial.print("probability = ");
  Serial.println(prob, 6);
  return true;
}

// ==========================================================
// VPT types & parameters (STRUCT DECLARED BEFORE FUNCTIONS)
// ==========================================================

struct VPTResult {
  int threshold_unit;    // amplitude unit 0..255, -1 on timeout
};

// ramp params
const uint8_t AMP_START   = 8;
const uint8_t AMP_MAX     = 250;
const uint8_t AMP_STEP    = 4;
const uint16_t STEP_MS    = 40;
const uint16_t SETTLE_MS  = 80;
const uint32_t RAMP_TIMEOUT_MS = 60000;  // safety cap in ms

// provisional amplitude->µm mapping (use dataset median as reference)
const float AMP_MEDIAN_OBS = 250.0f; // provisional observed median amplitude
const float AMP_TO_UM = IMP_MEDIAN[0] / AMP_MEDIAN_OBS; // µm per amplitude unit

// prototype
VPTResult runVPTThresholdRamp();

// Ramps amplitude in REALTIME mode until response or timeout
VPTResult runVPTThresholdRamp() {
  VPTResult res; res.threshold_unit = -1;
  if (!drvInitialized) {
    Serial.println("DRV2605 not initialized; cannot run threshold ramp.");
    return res;
  }

  // ensure reset released
  showTextPage("VPT ramp", "Release RESET to begin");
  while (digitalRead(BTN_RST_PIN) == LOW) delay(10);
  delay(20);

  // clear stale events
  for (int i = 0; i < NUM_BUTTONS; ++i) reportedPress[i] = false;

  // realtime mode
  drv.setMode(DRV2605_MODE_REALTIME);
  drv.setRealtimeValue(0);
  delay(SETTLE_MS);

  unsigned long rampStartMs = millis();
  unsigned long deadline = rampStartMs + RAMP_TIMEOUT_MS;
  uint8_t amp = AMP_START;
  drv.setRealtimeValue(amp);
  unsigned long lastStep = millis();

  while (true) {
    if (digitalRead(BTN_RST_PIN) == LOW) {
      drv.setRealtimeValue(0);
      drv.setMode(DRV2605_MODE_INTTRIG);
      res.threshold_unit = (int)amp;
      Serial.printf("VPT THRESH recorded: amp=%d\n", res.threshold_unit);
      return res;
    }

    if (millis() >= deadline) {
      drv.setRealtimeValue(0);
      drv.setMode(DRV2605_MODE_INTTRIG);
      Serial.println("VPT ramp timeout (no response)");
      res.threshold_unit = -1;
      return res;
    }

    unsigned long now = millis();
    if (now - lastStep >= STEP_MS) {
      lastStep = now;
      if (amp < AMP_MAX) {
        uint16_t next = (uint16_t)amp + AMP_STEP;
        amp = (next > AMP_MAX) ? AMP_MAX : (uint8_t)next;
        drv.setRealtimeValue(amp);
      } else {
        drv.setRealtimeValue(AMP_MAX);
      }
    }

    delay(1);
  }
}

// -------------------- Setup --------------------
void setup() {
  Serial.begin(115200);
  delay(50);
  Serial.println();
  Serial.println("CIPN TinyML + VPT (fixed) starting");

  // buttons
  initButtons();

  // OLED + I2C
  Wire.begin(); // default SDA/SCL for ESP32
  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
    Serial.println("SSD1306 allocation failed");
    for (;;) delay(1000);
  }
  display.display();
  delay(100);
  display.clearDisplay();
  showTextPage("CIPN TinyML", "Press 1..4 to answer", "RESET = restart/response");

  // drv init
  pinMode(DRV_EN_PIN, OUTPUT);
  digitalWrite(DRV_EN_PIN, HIGH);
  if (!drv.begin()) {
    Serial.println("ERROR: DRV2605 not found!");
    drvInitialized = false;
  } else {
    drvInitialized = true;
    drv.selectLibrary(1);
    drv.setMode(DRV2605_MODE_INTTRIG);
    drv.setRealtimeValue(0);
    Serial.println("DRV2605 ready");
  }

  // TF model
  tf.setNumInputs(N_FEATURES);
  tf.setNumOutputs(1);
  tf.resolver.AddFullyConnected();
  tf.resolver.AddRelu();
  tf.resolver.AddLogistic();
  tf.resolver.AddAdd();
  tf.resolver.AddSoftmax();

  Serial.println("Initializing model...");
  Eloquent::Error::Exception &exc = tf.begin(model_tflite);
  if (!exc.isOk()) {
    Serial.print("tf.begin failed: ");
    Serial.println(exc.toString());
    char buf[64];
    String s = exc.toString();
    s.toCharArray(buf, sizeof(buf));
    showTextPage("Model init failed", buf);
    while (true) delay(1000);
  }
  Serial.println("Model initialized OK.");

  memset(answers, 0, sizeof(answers));
  current_question = 0;
  finished = false;
  delay(300);
}

// -------------------- Loop --------------------
void loop() {
  updateButtons();

  if (!finished) {
    for (; current_question < NUM_QUESTIONS; ++current_question) {
      waitForChoiceAndStore(current_question);
    }
    finished = true;
    float s1 = sum_group_answers(G_SENSORY);
    float s2 = sum_group_answers(G_MOTOR);
    float s3 = sum_group_answers(G_AUTONOMIC);
    char buf[64];
    snprintf(buf, sizeof(buf), "S:%d M:%d A:%d", (int)s1, (int)s2, (int)s3);
    showTextPage("All done.", buf, "Press any answer to VPT+infer");
  } else {
    int ev = consumePressEvent();
    if (ev >= 0) {
      if (ev <= 3) {
        // any option triggers VPT test + inference
        prepare_and_run_model();
        showTextPage("Inference done", "Press RESET to retake");
        // wait for RESET to restart
        while (true) {
          updateButtons();
          int ev2 = consumePressEvent();
          if (ev2 == 4) break;
          delay(10);
        }
        memset(answers, 0, sizeof(answers));
        current_question = 0;
        finished = false;
        showTextPage("Restarting...", "", "");
        delay(300);
      } else if (ev == 4) {
        // reset pressed while finished -> immediate restart
        memset(answers, 0, sizeof(answers));
        current_question = 0;
        finished = false;
        showTextPage("Restarted", "", "");
        delay(300);
      }
    }
    delay(10);
  }

  delay(5);
}

// -------------------- Inference implementation --------------------
float realistic_cold_placeholder() {
  // small jitter around training mean
  int jitter = random(-15, 16); // -1.5 .. +1.5
  return SCALER_MEAN[1] + ((float)jitter / 10.0f);
}

void prepare_and_run_model() {
  float f_raw[N_FEATURES];
  f_raw[2] = sum_group_answers(G_SENSORY);
  f_raw[3] = sum_group_answers(G_MOTOR);
  f_raw[4] = sum_group_answers(G_AUTONOMIC);

  VPTResult vpt = runVPTThresholdRamp();
  float vpt_feature_raw;
  if (vpt.threshold_unit < 0) {
    vpt_feature_raw = IMP_MEDIAN[0];
  } else {
    vpt_feature_raw = (float)vpt.threshold_unit * AMP_TO_UM; // convert to µm
  }

  float cold_rand = realistic_cold_placeholder();

  f_raw[0] = vpt_feature_raw;
  f_raw[1] = cold_rand;

  Serial.print("VPT raw: threshold_unit=");
  Serial.print(vpt.threshold_unit);
  Serial.print(", feature_vpt_raw=");
  Serial.println(f_raw[0], 4);
  Serial.print("Cold raw (placeholder) = ");
  Serial.println(f_raw[1], 4);

  float f_norm[N_FEATURES];
  for (int i = 0; i < N_FEATURES; ++i) f_norm[i] = normalize_feature(i, f_raw[i]);

  Serial.println("Raw features (val -> norm):");
  for (int i = 0; i < N_FEATURES; ++i) {
    Serial.print(i); Serial.print(": ");
    Serial.print(f_raw[i]); Serial.print(" -> ");
    Serial.println(f_norm[i], 6);
  }

  run_inference_and_show(f_norm);
}
