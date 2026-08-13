#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.cc"
#include "chatbot_data.h"   // vocabulary[], intents[], VOCAB_SIZE, INTENT_COUNT,
                             // response_table[], response_counts[]

// Doubles the loop() task's stack (default 8 KB). Extra safety margin now
// that WiFi + WebServer + inference all run in that same task.
SET_LOOP_TASK_STACK_SIZE(16 * 1024);

// ============================================================
// WIFI CONFIG  <-- EDIT THESE TWO LINES ONLY
// ============================================================
const char* WIFI_SSID     = "europa";
const char* WIFI_PASSWORD = "Sajib11101992";

WebServer server(80);

// ============================================================
// MODEL
// ============================================================
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Measured actual usage was 39,772 bytes (see "Arena used" in Serial
// output) — 48 KB gives headroom without wasting RAM WiFi needs.
constexpr int kTensorArenaSize = 48 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// ============================================================
// WEB PAGE
// ============================================================
const char INDEX_HTML[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ESP32-C3 AI Chat</title>
<style>
  body { font-family: Arial, sans-serif; background:#1e1e2f; color:#eee; margin:0; padding:0; display:flex; flex-direction:column; height:100vh; }
  header { background:#2b2b40; padding:14px; text-align:center; font-size:18px; font-weight:bold; }
  #chat { flex:1; overflow-y:auto; padding:12px; }
  .msg { margin:8px 0; padding:8px 12px; border-radius:10px; max-width:75%; }
  .user { background:#3b82f6; margin-left:auto; text-align:right; }
  .bot { background:#3a3a52; margin-right:auto; }
  #inputBar { display:flex; padding:10px; background:#2b2b40; }
  #msgInput { flex:1; padding:10px; border:none; border-radius:6px; margin-right:8px; }
  #sendBtn { padding:10px 16px; border:none; border-radius:6px; background:#3b82f6; color:white; cursor:pointer; }
</style>
</head>
<body>
<header>ESP32-C3 Offline AI Chat</header>
<div id="chat"></div>
<div id="inputBar">
  <input id="msgInput" type="text" placeholder="Type a message..." onkeydown="if(event.key==='Enter') sendMsg()">
  <button id="sendBtn" onclick="sendMsg()">Send</button>
</div>
<script>
function addMsg(text, cls) {
  const chat = document.getElementById('chat');
  const div = document.createElement('div');
  div.className = 'msg ' + cls;
  div.textContent = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

function sendMsg() {
  const input = document.getElementById('msgInput');
  const text = input.value.trim();
  if (!text) return;
  addMsg(text, 'user');
  input.value = '';
  fetch('/chat?msg=' + encodeURIComponent(text))
    .then(r => r.text())
    .then(reply => addMsg(reply, 'bot'))
    .catch(() => addMsg('Error contacting device.', 'bot'));
}
</script>
</body>
</html>
)rawliteral";

// ============================================================
// FIND WORD IN VOCABULARY
// ============================================================
int findWord(const String& word) {
  for (int i = 0; i < VOCAB_SIZE; i++) {
    if (word == vocabulary[i]) return i;
  }
  return -1;
}

// ============================================================
// TOKENIZE + CREATE ONE-HOT SEQUENCE
// ============================================================
void encodeSequence(const String& text, float* sequence) {
  for (int i = 0; i < MAX_LENGTH * VOCAB_SIZE; i++) {
    sequence[i] = 0.0f;
  }

  int wordIndex = 0;
  String currentWord = "";

  for (int i = 0; i <= text.length() && wordIndex < MAX_LENGTH; i++) {
    char c = i < text.length() ? toLowerCase(text[i]) : ' ';

    if (isAlphaNumeric(c)) {
      currentWord += c;
    } else if (currentWord.length() > 0) {
      int vIndex = findWord(currentWord);
      if (vIndex >= 0) {
        sequence[(wordIndex * VOCAB_SIZE) + vIndex] = 1.0f;
      }
      wordIndex++;
      currentWord = "";
    }
  }
}

// ============================================================
// FIND MAX INTENT
// ============================================================
int findBestIntent(float* confidence) {
  int bestIndex = 0;
  float bestScore = -999.0f;

  for (int i = 0; i < INTENT_COUNT; i++) {
    float score = output->data.f[i];
    if (score > bestScore) {
      bestScore = score;
      bestIndex = i;
    }
  }

  *confidence = bestScore;
  return bestIndex;
}

// ============================================================
// RESPONSE
// ============================================================
String getResponse(int intent, float confidence) {
  if (confidence < 0.40f) {
    return "I'm not sure what you mean.";
  }

  int count = response_counts[intent];
  if (count <= 0) {
    return "I don't know.";
  }

  int pick = random(count);
  return String(response_table[intent][pick]);
}

// ============================================================
// RUN MODEL
// ============================================================
String runChatbot(String text, float* outConfidence, int* outIntent) {
  // static: this array is ~30 KB (12 * VOCAB_SIZE floats). On a normal
  // (non-static) local it would be allocated on the loop task's stack
  // EVERY call and overflow it (default stack is only 8 KB) -> crash.
  // static moves it to fixed memory allocated once, not per call.
  static float sequence[MAX_LENGTH * VOCAB_SIZE];
  encodeSequence(text, sequence);

  for (int i = 0; i < MAX_LENGTH * VOCAB_SIZE; i++) {
    input->data.f[i] = sequence[i];
  }

  TfLiteStatus status = interpreter->Invoke();

  if (status != kTfLiteOk) {
    return "ERROR: Inference failed!";
  }

  float confidence = 0.0f;
  int intent = findBestIntent(&confidence);

  if (outConfidence) *outConfidence = confidence;
  if (outIntent) *outIntent = intent;

  return getResponse(intent, confidence);
}

// ============================================================
// WEB HANDLERS
// ============================================================
void handleRoot() {
  server.send_P(200, "text/html", INDEX_HTML);
}

void handleChat() {
  if (!server.hasArg("msg")) {
    server.send(400, "text/plain", "Missing msg parameter");
    return;
  }

  String text = server.arg("msg");
  text.trim();

  if (text.length() == 0) {
    server.send(200, "text/plain", "Say something!");
    return;
  }

  float confidence = 0.0f;
  int intent = 0;
  String reply = runChatbot(text, &confidence, &intent);

  server.send(200, "text/plain", reply);
}

void handleNotFound() {
  server.send(404, "text/plain", "Not found");
}

// ============================================================
// SETUP
// ============================================================
void setup() {
  Serial.begin(115200);
  delay(1500);

  Serial.println();
  Serial.println("================================");
  Serial.println("       ESP32-C3 AI CHATBOT");
  Serial.println("================================");

  randomSeed(esp_random());

  Serial.println("Loading INT8 1D-CNN model...");

  model = tflite::GetModel(model_data);

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("ERROR: Model schema mismatch!");
    while (true) delay(1000);
  }

  static tflite::MicroMutableOpResolver<13> resolver;
  resolver.AddGather();
  resolver.AddConv2D();
  resolver.AddMean();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddQuantize();
  resolver.AddDequantize();
  resolver.AddReshape();
  resolver.AddLess();
  resolver.AddAdd();
  resolver.AddSelectV2();
  resolver.AddExpandDims();

  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize
  );

  interpreter = &static_interpreter;

  TfLiteStatus status = interpreter->AllocateTensors();

  if (status != kTfLiteOk) {
    Serial.println("ERROR: AllocateTensors failed!");
    Serial.println("Try increasing kTensorArenaSize, or check available heap below.");
    Serial.printf("Free heap: %d bytes\n", ESP.getFreeHeap());
    while (true) delay(1000);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("Model loaded successfully!");
  Serial.print("Input elements: ");
  Serial.println(input->bytes / sizeof(float));
  Serial.print("Output elements: ");
  Serial.println(output->bytes / sizeof(float));
  Serial.printf("Arena used: %u / %u bytes\n", interpreter->arena_used_bytes(), kTensorArenaSize);
  Serial.print("Free heap after AllocateTensors: ");
  Serial.println(ESP.getFreeHeap());

  // ----------------------------------------------------------
  // Connect to WiFi
  // ----------------------------------------------------------
  Serial.println();
  Serial.print("Connecting to WiFi: ");
  Serial.println(WIFI_SSID);

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 40) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println();
    Serial.println("WiFi connected!");
    Serial.print("Open this in your browser: http://");
    Serial.println(WiFi.localIP());
    Serial.print("Free heap after WiFi connect: ");
    Serial.println(ESP.getFreeHeap());
  } else {
    Serial.println();
    Serial.println("WiFi FAILED to connect. Serial chat still works below.");
  }

  server.on("/", handleRoot);
  server.on("/chat", handleChat);
  server.onNotFound(handleNotFound);
  server.begin();

  Serial.println();
  Serial.println("Chatbot ready (Serial + Web).");
}

// ============================================================
// LOOP
// ============================================================
void loop() {
  server.handleClient();

  if (Serial.available()) {
    String text = Serial.readStringUntil('\n');
    text.trim();

    if (text.length() == 0) return;

    unsigned long startTime = micros();

    float confidence = 0.0f;
    int intent = 0;
    String reply = runChatbot(text, &confidence, &intent);

    unsigned long inferenceTime = micros() - startTime;

    Serial.println();
    Serial.print("You: ");
    Serial.println(text);
    Serial.print("Intent: ");
    Serial.println(intents[intent]);
    Serial.print("Confidence: ");
    Serial.println(confidence, 3);
    Serial.print("Inference: ");
    Serial.print(inferenceTime);
    Serial.println(" us");
    Serial.print("Bot: ");
    Serial.println(reply);
    Serial.println();
  }
}