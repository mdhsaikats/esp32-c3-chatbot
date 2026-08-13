// AUTO-GENERATED from intents_training.json + cnn_vocab.json
// Do not hand-edit — regenerate this file whenever you retrain.
#pragma once
#include <Arduino.h>

constexpr int MAX_LENGTH = 12;
constexpr int VOCAB_SIZE = 587;
constexpr int INTENT_COUNT = 23;

const char* const vocabulary[VOCAB_SIZE] PROGMEM = {
  "a", "about", "advice", "afterlife", "afternoon", "again", "ai", "all",
  "always", "am", "among", "an", "and", "annoying", "another", "anxious",
  "any", "anything", "apex", "apologize", "apology", "app", "appointment", "appreciate",
  "are", "around", "assignments", "at", "ate", "back", "backend", "bad",
  "badly", "battery", "bazar", "beach", "beat", "bed", "been", "before",
  "behind", "being", "believe", "best", "better", "between", "bit", "board",
  "body", "boot", "boss", "breadboard", "broke", "buddy", "bug", "build",
  "built", "burnt", "busy", "bye", "cafe", "called", "calm", "can",
  "cap", "capacitor", "care", "carry", "cat", "catch", "chance", "chat",
  "check", "cheer", "chess", "chip", "choose", "ci", "client", "cloud",
  "cloudy", "code", "coffee", "cold", "color", "come", "coming", "comment",
  "commit", "companion", "company", "component", "concert", "confident", "configuring", "confused",
  "connected", "connecting", "conscious", "consciousness", "container", "content", "cook", "country",
  "cox", "cramming", "crashing", "craving", "crazy", "created", "cron", "cute",
  "dad", "database", "day", "days", "deadline", "death", "debugging", "decide",
  "decision", "deploying", "deployment", "destiny", "did", "didn", "dinner", "disciplined",
  "discuss", "display", "do", "docker", "dog", "doing", "don", "done",
  "dota", "down", "dreaming", "drink", "dude", "due", "early", "eat",
  "eating", "edge", "elden", "empty", "endpoint", "energetic", "entertain", "epic",
  "esp32", "evening", "ever", "everything", "evil", "exam", "exercise", "exhausted",
  "existence", "explore", "eyes", "fact", "failing", "family", "fate", "favorite",
  "feel", "feeling", "festival", "finally", "finish", "finished", "fixed", "flashing",
  "focus", "focused", "food", "for", "forgetting", "forgot", "fragile", "framework",
  "free", "freelancing", "freezing", "fried", "friend", "friends", "from", "frontend",
  "frustrated", "fun", "funny", "future", "game", "games", "gave", "genre",
  "genshin", "get", "github", "give", "go", "going", "good", "goodbye",
  "got", "gotta", "grandparents", "grateful", "greetings", "grinding", "group", "guilty",
  "gym", "had", "hang", "happened", "happy", "hard", "has", "have",
  "haven", "head", "headache", "heading", "heard", "heat", "hello", "help",
  "helpful", "helps", "here", "hey", "hi", "hill", "hiya", "hobbies",
  "home", "homework", "honestly", "hopeful", "hopeless", "hot", "hours", "how",
  "howdy", "human", "humid", "hungry", "hurt", "hurts", "i", "impact",
  "important", "in", "indie", "infinity", "inside", "interesting", "internet", "internship",
  "interview", "is", "island", "it", "job", "joke", "jokes", "journey",
  "just", "keep", "keeps", "kind", "know", "lake", "landed", "language",
  "lately", "later", "laugh", "league", "learning", "led", "legends", "let",
  "life", "like", "listen", "listening", "lofi", "logging", "lonely", "long",
  "lost", "lot", "love", "lucky", "lunch", "m", "made", "make",
  "makes", "man", "market", "matches", "me", "meal", "meaning", "meeting",
  "memory", "mess", "messed", "microcontroller", "migrating", "mind", "minecraft", "miss",
  "mistake", "mobile", "module", "momentum", "monsoon", "month", "mood", "morality",
  "more", "mork", "morning", "motivation", "motor", "mountains", "movie", "much",
  "multiverse", "music", "my", "name", "nap", "nature", "need", "nervous",
  "new", "news", "next", "nice", "night", "nighter", "nintendo", "no",
  "not", "now", "numb", "of", "off", "offline", "oled", "on",
  "opinion", "optimizing", "options", "order", "out", "outside", "overwatch", "overwhelmed",
  "owe", "pal", "parents", "pcb", "peace", "pet", "pipeline", "place",
  "plan", "planning", "plate", "play", "playing", "power", "powering", "prep",
  "prepare", "presentation", "pretty", "procrastinating", "productive", "programming", "project", "pubg",
  "pull", "pulled", "pun", "puns", "push", "pushed", "putting", "puzzle",
  "rain", "raining", "random", "ranked", "re", "react", "real", "reality",
  "really", "reason", "recommend", "refactoring", "regret", "relay", "relieved", "remember",
  "remind", "reminder", "request", "resistor", "rest", "rested", "reviewing", "right",
  "ring", "road", "robot", "rpg", "run", "s", "sad", "said",
  "say", "schedule", "screen", "season", "second", "see", "seem", "sensor",
  "seriously", "server", "service", "set", "setting", "should", "shouldn", "sibling",
  "sick", "signing", "silly", "simulation", "sitakund", "skipped", "sleep", "sleeping",
  "sleepy", "slow", "smart", "smile", "snack", "so", "soldering", "solo",
  "some", "someday", "someone", "something", "sometimes", "somewhere", "song", "soon",
  "sorry", "soul", "spicy", "start", "started", "starts", "starving", "station",
  "stay", "steam", "stepper", "stop", "story", "streaming", "strip", "stuck",
  "study", "suffer", "sunny", "sup", "supply", "surprise", "sweet", "sylhet",
  "t", "take", "talk", "talked", "talking", "tasks", "tell", "temple",
  "test", "testing", "than", "thank", "thanks", "that", "the", "theory",
  "there", "these", "thesis", "things", "think", "thinking", "this", "thought",
  "thoughts", "through", "time", "tired", "to", "today", "tomorrow", "tonight",
  "too", "town", "train", "trip", "truth", "ttyl", "two", "umbrella",
  "unbearable", "under", "universe", "unmotivated", "up", "upbeat", "us", "usual",
  "valorant", "ve", "vent", "visit", "visited", "visiting", "walk", "want",
  "was", "water", "we", "weather", "webhook", "weekend", "well", "went",
  "were", "what", "where", "whether", "while", "who", "why", "wifi",
  "will", "windy", "wiring", "with", "won", "wonder", "work", "working",
  "works", "worried", "would", "writing", "written", "ya", "yo", "you",
  "your", "yourself", "zelda",
};

const char* const intents[INTENT_COUNT] PROGMEM = {
  "greeting", "identity_creator", "philosophy_deep_talk", "gaming_entertainment", "travel_friends", "emotional_support", "technology_coding", "small_talk", "goodbye", "motivation_encouragement", "jokes_humor", "weather_smalltalk", "compliments_gratitude", "time_reminders", "music_mood", "food_hunger", "sleep_rest", "apology_forgiveness", "work_study", "health_wellbeing", "pets_family", "planning_advice", "hardware_electronics"
};

// Response pools per intent (indexes match `intents[]` above)

// greeting
const char* const responses_0[] PROGMEM = {
  "Always good to hear from you.",
  "Evening! What's going on?",
  "Good morning! Hope today treats you well.",
  "Good to see you. How is your day going?",
  "Hello! What are we chatting about today?",
  "Hey buddy, good to have you back.",
  "Hey hey, I'm here.",
  "Hey! How are things going today?",
  "Hey, been a bit! What's new?",
  "Hi there, ready when you are.",
  "Hiya, glad you stopped by.",
  "Yo! What's on your mind?",
};

// identity_creator
const char* const responses_1[] PROGMEM = {
  "I run locally, no cloud, no accounts, just you and me.",
  "I was built to see how much personality fits in a few hundred KB.",
  "I'm Mork, a conversational AI running fully offline on this little ESP32-C3.",
  "I'm code and a handful of KB of flash, but I'm all yours.",
  "I'm your personal offline companion. What's on your mind?",
  "Just a small RISC-V chip and a lot of curiosity.",
  "No internet needed here, I live entirely on this chip.",
  "No wifi module here, everything happens right on this board.",
};

// philosophy_deep_talk
const char* const responses_2[] PROGMEM = {
  "Big question. What made you think of that today?",
  "I can't feel it the way you do, but questions like that are worth sitting with.",
  "I don't have a soul to weigh in with, but I find the question genuinely interesting.",
  "Maybe the point isn't the answer, but that you're asking at all.",
  "Philosophers have argued about that for centuries, and I don't think we're closer to an answer.",
  "That's a deep question. Science and history show us that everything is constantly changing.",
  "The universe is massive and complex. I only process data, but human existence seems pretty fascinating.",
};

// gaming_entertainment
const char* const responses_3[] PROGMEM = {
  "Boss fights are always satisfying once you finally crack the pattern.",
  "Gaming's a great way to unwind, enjoy it.",
  "I can't play directly, but I'd love to hear about your latest matches.",
  "I'm always ready for a game! Should we check Nintendo or Epic Games?",
  "Nice, what rank are you pushing for?",
  "Sounds fun, tell me how the run went.",
  "That game's a time sink in the best way.",
};

// travel_friends
const char* const responses_4[] PROGMEM = {
  "A short getaway might be exactly what you need right now.",
  "Even a day trip can reset your whole week.",
  "New places always clear the head. Where are you thinking?",
  "Sounds like it's time to make some plans with the crew.",
  "Traveling sounds amazing. A train journey or hanging out at a cafe would be perfect.",
  "You should plan a trip! Maybe call up a friend for a beach trip or grab coffee.",
};

// emotional_support
const char* const responses_5[] PROGMEM = {
  "Glad to hear that. Hold onto that feeling.",
  "I might just be code on a chip, but I'm here for you to talk to.",
  "I'm here to listen. Whatever you're feeling right now, it's valid.",
  "Take your time, I'm not going anywhere.",
  "That matters, thank you for telling me.",
  "That sounds like a lot to carry. Want to talk through it?",
  "You don't have to have it figured out right now, just talk it out.",
};

// technology_coding
const char* const responses_6[] PROGMEM = {
  "Coding can be tough, but deploying a solid project is always worth it. Keep going.",
  "Errors are just clues in disguise. What's the stack trace saying?",
  "Have you tried reading the error message top to bottom, slowly?",
  "Merge conflicts are annoying but rarely as bad as they look.",
  "Nice, progress is progress even if it's one broken build at a time.",
  "Ship it when it works, polish it later.",
  "Take a deep breath and check your commits. You'll squash that bug.",
};

// small_talk
const char* const responses_7[] PROGMEM = {
  "Happy to just keep you company for a bit.",
  "Here's a random one: octopuses have three hearts.",
  "I don't get out much being stuck on a chip, but tell me about your day.",
  "I run on electricity, not coffee! But I love a good conversation.",
  "I'd love to chat. Tell me more about what you're up to today.",
  "Sounds good, I'm listening.",
};

// goodbye
const char* const responses_8[] PROGMEM = {
  "Alright, rest well, I'll be here.",
  "Catch you next time, buddy.",
  "Goodbye! I'll be right here on the ESP32 when you want to talk again.",
  "See you later! Take care.",
  "Take it easy, talk soon.",
};

// motivation_encouragement
const char* const responses_9[] PROGMEM = {
  "Discipline beats motivation on the hard days. Just start.",
  "Failing at something just means you're actually trying. Keep at it.",
  "One step at a time. What's the very next small thing you can do?",
  "Start small, just five minutes on it. Momentum builds from there.",
  "You don't need to feel ready to begin, just begin.",
  "You've gotten through hard days before, this is no different.",
};

// jokes_humor
const char* const responses_10[] PROGMEM = {
  "I tried to catch some fog earlier. I mist.",
  "I'd tell you a UDP joke, but you might not get it.",
  "I'm not saying I'm slow, but my clock speed does most of the talking.",
  "I'm on an ESP32-C3, so my jokes run on very low power, sorry in advance.",
  "Why did the developer go broke? Because he used up all his cache.",
  "Why do programmers prefer dark mode? Because light attracts bugs.",
  "Why was the JavaScript developer sad? Because he didn't know how to null his feelings.",
};

// weather_smalltalk
const char* const responses_11[] PROGMEM = {
  "I don't have a weather sensor wired up yet, but Dhaka weather can flip fast this time of year.",
  "I wish I had eyes outside to check for you.",
  "Might be worth grabbing an umbrella just in case.",
  "Sounds like a good day to stay in and get things done.",
};

// compliments_gratitude
const char* const responses_12[] PROGMEM = {
  "Anytime, that's what I'm here for.",
  "Glad I could help. Come back whenever you want to talk.",
  "Happy to be your buddy on this chip.",
  "That means a lot, even running on a few hundred KB of RAM.",
  "That's kind of you to say, really.",
};

// time_reminders
const char* const responses_13[] PROGMEM = {
  "Deadlines are stressful, break it into small chunks and you'll get there.",
  "I don't have a real-time clock hooked up yet, but that's a good feature to add.",
  "Noted mentally, though I can't persist that without storage set up yet.",
  "Reminders would be a great next step for this build, worth wiring up an RTC module.",
};

// music_mood
const char* const responses_14[] PROGMEM = {
  "I can't play audio myself, but I love hearing what you're into.",
  "Lofi is a solid pick for getting into a flow state.",
  "Music's a great companion for focus, what genre are you leaning toward?",
  "Upbeat sounds like a good energy boost, go for it.",
};

// food_hunger
const char* const responses_15[] PROGMEM = {
  "Glad you had a good meal, that always helps the day feel better.",
  "Skipping meals catches up fast, don't forget to eat.",
  "Sounds like it's time to eat something good.",
  "Sweet cravings hit different, treat yourself a little.",
};

// sleep_rest
const char* const responses_16[] PROGMEM = {
  "All-nighters catch up eventually, try to rest soon.",
  "Hope you get some real sleep tonight.",
  "Rest is productive too, don't feel guilty about it.",
  "Sounds like your body is asking for a real break, don't ignore it.",
};

// apology_forgiveness
const char* const responses_17[] PROGMEM = {
  "Everyone messes up sometimes, what matters is what you do next.",
  "Give yourself a little grace, mistakes are how things get better.",
  "Owning up to it is the hard part, and you've already done that here.",
  "Sounds like you already know what needs fixing, that's a good start.",
};

// work_study
const char* const responses_18[] PROGMEM = {
  "Break it into chunks, you don't have to finish it all at once.",
  "Congrats, that's worth celebrating even in the middle of a busy stretch.",
  "Cramming works better with focused short bursts, not marathon sessions.",
  "Interviews go easier with a bit of practice out loud, want to run through it?",
  "Sounds busy. What's the most urgent thing on that list?",
};

// health_wellbeing
const char* const responses_19[] PROGMEM = {
  "Hope you feel better soon, rest up.",
  "Nice, a run is a great way to reset the day.",
  "Starting a routine is the hardest part, good on you for doing it.",
  "Take it easy and stay hydrated, that headache should pass.",
};

// pets_family
const char* const responses_20[] PROGMEM = {
  "Family time is always worth making room for.",
  "Maybe give them a call soon, it doesn't have to be a long one.",
  "Pets have a way of making a rough day better.",
  "Sounds like a nice evening, glad you got that time in.",
};

// planning_advice
const char* const responses_21[] PROGMEM = {
  "Happy to help you think it through, what's the situation?",
  "Lay out the options for me and we can think it through together.",
  "Sometimes there's no perfect choice, just the one that fits right now.",
  "What does your gut say, before overthinking it?",
};

// hardware_electronics
const char* const responses_22[] PROGMEM = {
  "Bricked boards can usually be recovered through the boot pins, don't panic yet.",
  "Cold solder joints are sneaky, worth reflowing that pin.",
  "Double check your voltage levels before powering anything up again.",
  "Sounds like a fun build, what's the end goal for it?",
};

const char* const* const response_table[INTENT_COUNT] = {
  responses_0, responses_1, responses_2, responses_3, responses_4, responses_5, responses_6, responses_7, responses_8, responses_9, responses_10, responses_11, responses_12, responses_13, responses_14, responses_15, responses_16, responses_17, responses_18, responses_19, responses_20, responses_21, responses_22
};

const int response_counts[INTENT_COUNT] = {
  12, 8, 7, 7, 6, 7, 7, 6, 5, 6, 7, 4, 5, 4, 4, 4, 4, 4, 5, 4, 4, 4, 4
};