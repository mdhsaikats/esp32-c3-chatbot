#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.cc"


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

constexpr int kTensorArenaSize = 20 * 1024;
uint8_t tensor_arena[kTensorArenaSize];


// ============================================================
// VOCABULARY
// ============================================================

const char* vocabulary[] = {
  "you", "what", "is", "do", "are", "can", "me", "a", "ai", "esp32",
  "good", "your", "tell", "need", "hello", "hey", "there", "see", "later", "talk",
  "to", "who", "name", "an", "help", "c3", "about", "thanks", "hi", "morning",
  "afternoon", "evening", "bye", "goodbye", "am", "i", "talking", "robot", "abilities", "with",
  "online", "internet", "wifi", "server", "use", "the", "cloud", "work", "offline", "microcontroller",
  "does", "mean", "artificial", "intelligence", "thank", "lot", "that", "helpful", "nice", "cool"
};

constexpr int VOCAB_SIZE = 60;


// ============================================================
// INTENTS
// ============================================================

const char* intents[] = {
  "greeting", "goodbye", "identity", "capabilities",
  "offline", "esp32", "ai", "thanks"
};

constexpr int INTENT_COUNT = 8;


// ============================================================
// QUANTIZATION
// ============================================================

constexpr float INPUT_SCALE = 0.003921568859368563f;
constexpr int INPUT_ZERO_POINT = -128;

constexpr float OUTPUT_SCALE = 0.00390625f;
constexpr int OUTPUT_ZERO_POINT = -128;


// ============================================================
// WEB PAGE (stored in flash)
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
// TOKENIZE + CREATE BOW VECTOR
// ============================================================

void createInputVector(const String& text, int8_t* outputVector) {
  for (int i = 0; i < VOCAB_SIZE; i++) {
    outputVector[i] = INPUT_ZERO_POINT;
  }

  String word = "";

  for (int i = 0; i <= text.length(); i++) {
    char c;
    if (i < text.length()) c = text[i];
    else c = ' ';

    if (c >= 'A' && c <= 'Z') c = c + ('a' - 'A');

    if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) {
      word += c;
    } else {
      if (word.length() > 0) {
        int index = findWord(word);
        if (index >= 0) {
          outputVector[index] = 127;
        }
        word = "";
      }
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
    float score = (static_cast<float>(output->data.int8[i]) - OUTPUT_ZERO_POINT) * OUTPUT_SCALE;
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

  switch (intent) {
    case 0: return "Hello! Nice to meet you.";
    case 1: return "Goodbye!";
    case 2: return "I am a tiny offline AI running on an ESP32-C3.";
    case 3: return "I can understand simple questions and identify your intent.";
    case 4: return "No. I work completely offline without WiFi or internet.";
    case 5: return "ESP32-C3 is a small RISC-V microcontroller.";
    case 6: return "AI means Artificial Intelligence.";
    case 7: return "You're welcome!";
    default: return "I don't know.";
  }
}


// ============================================================
// RUN MODEL ON A MESSAGE -> RETURNS BOT REPLY
// (shared by Serial and Web)
// ============================================================

String runChatbot(String text, float* outConfidence, int* outIntent) {
  int8_t inputVector[VOCAB_SIZE];
  createInputVector(text, inputVector);

  for (int i = 0; i < VOCAB_SIZE; i++) {
    input->data.int8[i] = inputVector[i];
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

  Serial.println("Loading INT8 model...");

  model = tflite::GetModel(model_data);

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("ERROR: Model schema mismatch!");
    while (true) delay(1000);
  }

  static tflite::MicroMutableOpResolver<8> resolver;
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddReshape();
  resolver.AddQuantize();
  resolver.AddDequantize();
  resolver.AddGather();

  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize
  );

  interpreter = &static_interpreter;

  TfLiteStatus status = interpreter->AllocateTensors();

  if (status != kTfLiteOk) {
    Serial.println("ERROR: AllocateTensors failed!");
    while (true) delay(1000);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("Model loaded successfully!");
  Serial.print("Input elements: ");
  Serial.println(input->bytes);
  Serial.print("Output elements: ");
  Serial.println(output->bytes);

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
  } else {
    Serial.println();
    Serial.println("WiFi FAILED to connect. Check SSID/password.");
    Serial.println("Chat will still work over Serial.");
  }

  // ----------------------------------------------------------
  // Start web server
  // ----------------------------------------------------------

  server.on("/", handleRoot);
  server.on("/chat", handleChat);
  server.onNotFound(handleNotFound);
  server.begin();

  Serial.println();
  Serial.println("Chatbot ready.");
  Serial.println("Type a message here, or open the web page:");
  Serial.println();
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

    if (text.equalsIgnoreCase("exit")) {
      Serial.println("Chatbot stopped.");
      while (true) {
        server.handleClient();
        delay(10);
      }
    }

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