#Andrew Gasiorowski


import sys, re
from sklearn import tree
from sklearn.feature_extraction import DictVectorizer

#opens a file and returns it's content as a list of tupples and num lines in file
def return_text_list(fileName):
    text_list = []
    num_lines = 0
    with open(fileName) as file:
        for line in file:
            content = line.split()
            text_list.append(content)
            num_lines = num_lines + 1
    return text_list, num_lines

#takes a list of tuples and returns a dictionary with features extracted and a class listing
def extract_features(text_list, num):
    feature_list = []
    class_list = []
    output_list = []
    #I assume that a period will never be the first 'char' in the list
    #Instances of form like 'J.B.' will be treated as 'JB.' though not transformed(period remains)
    for line in range(num-1):
        if text_list[line][2][0] != 'T':
            #The following lines define the contents of the dictionary
            #[strip period from word on lhs of period,
            #don't remove any punction from word on rhs of period,
            #binary: 1 if lhs<3,
            #bianry: 1 if lhs first char capitalized,
            #binary: 1 if rhs first char capitalized
            #binary: 1 if word to the right of "." ended with "."
            #binary: 1 if length of R < 3
            #binary: 1 if word to left of "." is followed by word ending with "."
            lhs_len = 1 if len(text_list[line][1][:-1])<3 else 0
            lhs_upper = 1 if text_list[line][1][0].isupper() == True else 0
            rhs_upper = 1 if text_list[line+1][1][0].isupper() == True else 0
            rhs_rhs_period = 1 if text_list[line+1][1][-1] == "." else 0
            rhs_len = 1 if len(text_list[line+1][1][:-1])<3 else 0
            lhs_lhs_period = 1 if text_list[line-1][1][-1] == "." else 0
            features = {'lhs_word':text_list[line][1][:-1], 'rhs_word':text_list[line+1][1], 'lhs_lt_3':lhs_len, 'lhs_uppercase':lhs_upper, 'rhs_uppercase':rhs_upper, 'rhs_rhs_upper':rhs_rhs_period, 'rhs_len':rhs_len, 'lhslhs_period':lhs_lhs_period}
            feature_list.append(features)
            class_list.append(text_list[line][2])
            output_list.append((text_list[line][1],text_list[line][2]))
    return feature_list, class_list, output_list
            
#takes a dictionary and class list as input returns a trained decision tree classifier
def train_classifier(x, y):
    vectorizer = DictVectorizer()
    x_train = vectorizer.fit_transform(x)
    classifier = tree.DecisionTreeClassifier()
    classifier = classifier.fit(x_train, y)
    return classifier, vectorizer

#takes a set of input features that match the requirements of the classifier argument
#returns a list of predictions on the input
def test_classifier(x, classifier, vectorizer):
    x_test = vectorizer.transform(x)
    result = classifier.predict(x_test)
    return result

#takes a set of predictions and a set of answers and computes accuracy
def compute_accuracy(y_actual, y_predict):
    correct = 0
    num_elements = len(y_actual)
    for i in range(num_elements):
        if y_actual[i] == y_predict[i]:
            correct = correct + 1
    return correct / num_elements

#creates output file
def create_output_file(testing_data, prediction):
    template = "{0:15}{1:5}{2:5}"
    f = open("SBD.test.out", "w")
    for i in range(len(testing_data)):
        line_tup = (testing_data[i][0],testing_data[i][1],prediction[i])
        f.write(template.format(*line_tup)+'\n')
    f.close()

#This block of code grabs file names from command line        
able = str(sys.argv).split(',')
regex = re.compile('[^a-zA-Z.]')
training_file = regex.sub('', able[1])
testing_file = regex.sub('', able[2])

#this block of code preprocesses training data and trains classifier
training_data, num_ele_train = return_text_list(training_file)
x_train, y_train,train_output_list = extract_features(training_data, num_ele_train)
decision_tree_classifier, vectorizer = train_classifier(x_train, y_train)

#this block of code preprocesses testing data and tests classifier
testing_data, num_ele_test = return_text_list(testing_file)
x_test, y_test, test_output_list = extract_features(testing_data, num_ele_test)
y_predict = test_classifier(x_test, decision_tree_classifier, vectorizer)
acc = compute_accuracy(y_test, y_predict)
print(str(round(acc, 4)))

#this outputs the results to a file
create_output_file(test_output_list, y_predict)
