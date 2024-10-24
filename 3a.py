import numpy as np
import matplotlib.pyplot as plt

	%matplotlib inline

	iq_scores = [126,  89,  90, 101, 102,  74,  93, 101,  66, 120, 108,  97,  98,
            105, 119,  92, 113,  81, 104, 108,  83, 102, 105, 111, 102, 107,
            103,  89,  89, 110,  71, 110, 120,  85, 111,  83, 122, 120, 102,
            84, 118, 100, 100, 114,  81, 109,  69,  97,  95, 106, 116, 109,
            114,  98,  90,  92,  98,  91,  81,  85,  86, 102,  93, 112,  76,
            89, 110,  75, 100,  90,  96,  94, 107, 108,  95,  96,  96, 114,
            93,  95, 117, 141, 115,  95,  86, 100, 121, 103,  66,  99,  96,
            111, 110, 105, 110,  91, 112, 102, 112,  75]

	plt.figure(figsize=(6, 4), dpi=150)
	plt.hist(iq_scores, bins=10)
	plt.axvline(x=100, color='r')
	plt.axvline(x=115, color='r', linestyle= '--')
	plt.axvline(x=85, color='r', linestyle= '--')

	plt.xlabel('IQ score')
plt.ylabel('Frequency')
plt.title('IQ scores for a test group of a hundred adults')
plt.show()


plt.figure(figsize=(6, 4), dpi=150)

plt.boxplot(iq_scores)

ax = plt.gca()
ax.set_xticklabels(['Test group'])
plt.ylabel('IQ score')
plt.title('IQ scores for a test group of a hundred adults')

plt.show()
group_a = [118, 103, 125, 107, 111,  96, 104,  97,  96, 114,  96,  75, 114,
       107,  87, 117, 117, 114, 117, 112, 107, 133,  94,  91, 118, 110,
       117,  86, 143,  83, 106,  86,  98, 126, 109,  91, 112, 120, 108,
       111, 107,  98,  89, 113, 117,  81, 113, 112,  84, 115,  96,  93,
       128, 115, 138, 121,  87, 112, 110,  79, 100,  84, 115,  93, 108,
       130, 107, 106, 106, 101, 117,  93,  94, 103, 112,  98, 103,  70,
       139,  94, 110, 105, 122,  94,  94, 105, 129, 110, 112,  97, 109,
       121, 106, 118, 131,  88, 122, 125,  93,  78]
group_b = [126,  89,  90, 101, 102,  74,  93, 101,  66, 120, 108,  97,  98,
            105, 119,  92, 113,  81, 104, 108,  83, 102, 105, 111, 102, 107,
            103,  89,  89, 110,  71, 110, 120,  85, 111,  83, 122, 120, 102,
            84, 118, 100, 100, 114,  81, 109,  69,  97,  95, 106, 116, 109,
            114,  98,  90,  92,  98,  91,  81,  85,  86, 102,  93, 112,  76,
            89, 110,  75, 100,  90,  96,  94, 107, 108,  95,  96,  96, 114,
            93,  95, 117, 141, 115,  95,  86, 100, 121, 103,  66,  99,  96,
            111, 110, 105, 110,  91, 112, 102, 112,  75]
group_c = [108,  89, 114, 116, 126, 104, 113,  96,  69, 121, 109, 102, 107,
       122, 104, 107, 108, 137, 107, 116,  98, 132, 108, 114,  82,  93,
        89,  90,  86,  91,  99,  98,  83,  93, 114,  96,  95, 113, 103,
        81, 107,  85, 116,  85, 107, 125, 126, 123, 122, 124, 115, 114,
        93,  93, 114, 107, 107,  84, 131,  91, 108, 127, 112, 106, 115,
        82,  90, 117, 108, 115, 113, 108, 104, 103,  90, 110, 114,  92,
       101,  72, 109,  94, 122,  90, 102,  86, 119, 103, 110,  96,  90,
       110,  96,  69,  85, 102,  69,  96, 101,  90]
group_d = [ 93,  99,  91, 110,  80, 113, 111, 115,  98,  74,  96,  80,  83,
       102,  60,  91,  82,  90,  97, 101,  89,  89, 117,  91, 104, 104,
       102, 128, 106, 111,  79,  92,  97, 101, 106, 110,  93,  93, 106,
       108,  85,  83, 108,  94,  79,  87, 113, 112, 111, 111,  79, 116,
       104,  84, 116, 111, 103, 103, 112,  68,  54,  80,  86, 119,  81,
        84,  91,  96, 116, 125,  99,  58, 102,  77,  98, 100,  90, 106,
       109, 114, 102, 102, 112, 103,  98,  96,  85,  97, 110, 131,  92,
        79, 115, 122,  95, 105,  74,  85,  85,  95]

plt.figure(figsize=(6, 4), dpi=150)

plt.boxplot([group_a, group_b, group_c, group_d])

ax = plt.gca()
ax.set_xticklabels(['Group A', 'Group B', 'Group C', 'Group D'])
plt.ylabel('IQ score')
plt.title('IQ scores for different test groups')

plt.show()
