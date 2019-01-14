import string
'''
Run this first on your file to tokenize your document as a set of sequences, 50 input words, 1 output
'''

# Load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text


# Load document, test it loaded
in_filename = 'readings.txt'
doc = load_doc(in_filename)
print(doc[:200])


def clean_doc(doc):
    """Turn a doc into a clean set of tokens
            Args:
                doc: any string - long file
            Returns:
                list of tokens
        """
    # replace '--' with a space ' '
    doc = doc.replace('.', ' <EOS>')
    doc = doc.replace('?', ' <QUES>')
    # split into tokens by white space
    tokens = doc.split()
    # remove punctuation from each token
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    return tokens


# Clean document
tokens = clean_doc(doc)
print(tokens[:200])
print('Total Tokens: %d' % len(tokens))
print('Unique Tokens: %d' % len(set(tokens)))


# Organize into sequences of tokens, with 50 input words and 1 output word
# We iterate over the list of tokens from token 51 onward and take the prior 50
# Transform the tokens into space-separated strings for later file storage
length = 50 + 1
sequences = list()
for i in range(length, len(tokens)):
	# select sequence of tokens
	seq = tokens[i-length:i]
	# convert into a line
	line = ' '.join(seq)
	# store
	sequences.append(line)
print('Total Sequences: %d' % len(sequences))


# Save tokens to file, one dialog per line
def save_doc(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()


# Save sequences to file
out_filename = 'readings_sequences.txt'
save_doc(sequences, out_filename)
