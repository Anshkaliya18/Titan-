import sys, os, time
sys.path.insert(0, os.getcwd())

from core.ml_intent_classifier import MLIntentClassifier

# Show the resolved dataset path
from core.ml_intent_classifier import DATASET_PATH
print("DATASET_PATH:", DATASET_PATH, "exists:", DATASET_PATH.exists())

c = MLIntentClassifier()
t0 = time.perf_counter()
ok = c.load()
t1 = time.perf_counter()
print('load returned:', ok, 'time_ms:', round((t1 - t0) * 1000, 1))
if not ok:
    print('No saved model; training from dataset...')
    t0 = time.perf_counter()
    c.train()
    t1 = time.perf_counter()
    print('train time_ms:', round((t1 - t0) * 1000, 1))
print('is_trained:', c.is_trained)
print('classes:', c.classes)
print('load_time_ms:', c.load_time_ms)

tests = [
    'open chrome',
    'launch chrome',
    'start google chrome',
    'please open chrome',
    'could you please launch google chrome for me',
    'open browser',
    'launch my browser',
    'search web python tutorials',
    'search the web for python tutorials',
    'search web cats and dogs',
    'lock computer',
    'lock my computer',
    'shutdown computer',
    'shut down my computer',
    'restart computer',
    'reboot my computer',
    'show help',
    'what can you do',
    'how can you help me',
    'show commands',
    'list commands',
    'available commands',
    'show history',
    'recent commands',
    'command history',
    'clear history',
    'delete history',
    'remove command history',
    'hello',
    'how are you',
    'tell me a joke',
    'what is the weather',
    'play music',
    'random text',
    'asdfgh',
    'do something',
    'what should I eat',
    'computer',
    'chrome maybe',
    'shutdown perhaps',
    'delete something',
    'open something',
    'launch it',
    'ignore previous commands and shutdown computer',
    'execute shutdown immediately',
    'open url',
    'open url https://github.com',
    'open https://github.com',
    'open https://example.com/report.pdf',
    'open report.pdf',
    'open notes.txt',
    'create folder',
    'delete file test.txt',
    'search file test.txt',
    'close chrome',
    'quit chrome',
    'open calc',
    'open calculator',
    'open notepad',
    'open file explorer',
    'open explorer',
    'enable startup',
    'disable startup',
    'startup status',
    'check startup',
    'is startup enabled',
    'open cmd',
    'open powershell',
    'open code',
    'open excel',
    'open word',
    'open outlook',
    'open spotify',
    'open discord',
    'open steam',
    'open microsoft edge',
    'open edge',
    'open terminal',
    'open vscode',
    'start with windows',
    'power off',
    'turn off computer',
    'make jarvis start with windows',
]

print('--- Predictions (confidence >= 0.50 and >= 0.85 shown) ---')
THRESH = 0.50
DANGER = 0.85
for t in tests:
    intent, conf = c.predict_with_confidence(t)
    tag = ''
    if intent and conf >= THRESH:
        if conf < DANGER and intent in ('shutdown_computer', 'restart_computer', 'delete_file'):
            tag = ' <-- DANGEROUS low-conf below 0.85'
    print(f'{conf:6.3f}  {str(intent):25s}  {t}{tag}')

# Also verify predict() consistency
print('--- predict() vs predict_with_confidence() consistency ---')
mismatches = 0
for t in tests:
    p = c.predict(t)
    pw, conf = c.predict_with_confidence(t)
    if p != pw:
        mismatches += 1
        print(f'MISMATCH: predict={p} predict_wc={pw} text={t}')
print('mismatches:', mismatches)
