import os
from timeit import default_timer as timer

def pre_process(train_input_path,validation_input_path,test_input_path, output_path):

    start_time = timer()

    if not os.path.exists(train_input_path):
        print("Error: invalid input path for pre_process")
        return

    if os.path.exists(output_path):
        print("Warning: output file already exists, pre_process will override")
    elif not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    vocabulary = {}
    with open(train_input_path, 'r') as input_file:
        lines = [line.rstrip('\n') for line in input_file]

    for line in lines:
        for word in line.split():
            if word in vocabulary:
                vocabulary[word] += 1
            else:
                vocabulary[word] = 1

    #sanity check (see that all the words in the train\validation\test are in the vocabulary
    files_to_check = [train_input_path,validation_input_path,test_input_path]
    for curr_path in files_to_check:
        with open(curr_path, 'r') as input_file:
            lines = [line.rstrip('\n') for line in input_file]

        for line in lines:
            if word not in vocabulary:
                print("Warning: the word \'",word,"\' from file: \'",os.path.basename(curr_path),"\' is not in the vocabulary")

    #write to file the vocabulary
    sorted_vocabulary = [(key, vocabulary[key]) for key in sorted(vocabulary, key=vocabulary.get, reverse=True)]

    with open(output_path,'w') as output_file:
        for key,value in sorted_vocabulary:
            output_line = str(key) + ' ' + str(value) +'\n'
            output_file.write(output_line)

    end_time = timer()

    print('\n##################################################################')
    print('PreProcess finished building vocabulary time: ',end_time-start_time,'[sec]')
    print('Input: ', train_input_path)
    print('Output: ',output_path)
    print('Vocabulary size: ',len(sorted_vocabulary))
    print('##################################################################\n')

if __name__ == "__main__":
    pre_process(os.path.abspath("ptb.train.txt"),
                os.path.abspath("ptb.valid.txt"),
                os.path.abspath("ptb.test.txt"),
                os.path.abspath("vocabulary.txt"))
