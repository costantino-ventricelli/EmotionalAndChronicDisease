# coding=utf-8

FILE_NAME = ["disease_training.txt", "healthy_training.txt", "disease_test.txt",
             "healthy_test.txt", "disease_validation.txt", "healthy_validation.txt"]

ON_SURFACE = 1
ON_AIR = 0

HEALTHY = 0
DISEASE = 1

# Hand task extension
CLOCK = "_cdt."
NATURAL_SENTENCE = "_sw."
PENTAGON = "_ipc."
MATRIX_1 = "_m1."
MATRIX_2 = "_m2."
MATRIX_3 = "_m3."
T_TRIAL_1 = "_tmtt1."
TRIAL_1 = "_tmt1."
T_TRIAL_2 = "_tmtt2."
TRIAL_2 = "_tmt2."
HELLO = "_h."
V_POINT = "_vp."
H_POINT = "_hp."
SQUARE = "_sc."
SIGNATURE_1 = "_s1."
SIGNATURE_2 = "_s2."
COPY_SPIRAL = "_cs."
TRACED_SPIRAL = "_ts."
BANK_CHECK = "_chk."
LE = "_le."
MOM = "_mom."
WINDOW = "_w."
LISTENING = "_ds."

# Emothaw
HOUSE = '_sc.'
FOUR_WORDS = '_fw.'
LEFT_RING = '_lr.'
RIGHT_RING = '_rr.'

TASKS = [CLOCK, NATURAL_SENTENCE, PENTAGON, MATRIX_1, MATRIX_2, MATRIX_3, TRIAL_1, T_TRIAL_1, T_TRIAL_2,
        TRIAL_2, HELLO, V_POINT, H_POINT, SQUARE, SIGNATURE_1, SIGNATURE_2, COPY_SPIRAL, TRACED_SPIRAL,
        BANK_CHECK, LE, MOM, WINDOW, LISTENING]

# Hand task name
TASKS_MAME = {
    "_cdt.": "CLOCK",
    "_sw.": "NATURAL_SENTENCE",
    "_ipc.": "PENTAGON",
    "_m1.": "MATRIX_1",
    "_m2.": "MATRIX_2",
    "_m3.": "MATRIX_3",
    "_tmtt1.": "T_TRIAL_1",
    "_tmt1.": "TRIAL_1",
    "_tmtt2.": "T_TRIAL_2",
    "_tmt2.": "TRIAL_2",
    "_h.": "HELLO",
    "_vp.": "V_POINT",
    "_hp.": "H_POINT",
    "_sc.": "SQUARE",
    "_s1.": "SIGNATURE_1",
    "_s2.": "SIGNATURE_2",
    "_cs.": "COPY_SPIRAL",
    "_ts.": "TRACED_SPIRAL",
    "_chk.": "BANK_CHECK",
    "_le.": "LE",
    "_mom.": "MOM",
    "_w.": "WINDOW",
    "_ds.": "LISTENING"
}

