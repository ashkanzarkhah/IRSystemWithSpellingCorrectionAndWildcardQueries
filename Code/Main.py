#Loading documents in memory from docs folder
import os

folder_path = 'docs'
files = os.listdir(folder_path)

documents = []
for file in files:
    file_path = os.path.join(folder_path, file)
    with open(file_path, 'r') as file:
        document = file.read()
        documents.append(document)

#First we lower_case our data
documents = [document.lower() for document in documents]

#Next we try to tokenize our documents with nltk library
import nltk
from nltk.tokenize import word_tokenize

tokenized_documents = []
for document in documents:
    tokenized_documents.append(word_tokenize(document))

#for tokenized_document in tokenized_documents:
#    print(tokenized_document)

#As we can see, we have some tokens with no meanings, now we try to find them
#commenly no meaning tokens are stop words or small tokens
#we try to find stop words with their repetiotion and small tokens by their length
#first we go for small tokens

small_tokens = []
for tokenized_document in tokenized_documents:
    for token in tokenized_document:
        if(len(token) <= 2):
            small_tokens.append(token)
small_tokens = list(set(small_tokens))
#print(small_tokens)

#Now befor going any further we realize a problem in our tokenizing,
#which is we have shortened words like i'm and i'd and i'll and don't and doesn't and ...
#so we go back to our first document and try to fix all the shortened words
#first we try to find them
import re

shortenedwords = []

for document in documents:
    shortenedwords.append(re.findall(r'\w+[\'’]\w*', document))
    
shortenedwords = [word for shortenedword in shortenedwords for word in shortenedword]
shortenedwords = list(set(shortenedwords))
#print(shortenedwords)

#Now we create a dictionary for shortened words
#And we put won't and can't at first because of their special form
shortened_words_dic = {
    'won’t' : ' will not',
    'can’t' : ' can not',
    '’ve' : ' have',
    '’m' : ' am',
    '’d' : ' would',
    '’ll' : ' will',
    '’s': ' is',
    '’re': ' are',
    'n’t' : ' not'
}

#Now we replace each of them with their long form 
new_documents = []
for document in documents:
    new_document = document
    for key, value in shortened_words_dic.items():
        new_document = new_document.replace(key, value)
    new_documents.append(new_document)

documents = new_documents

#Now we try tokenizing and finding short tokens again
tokenized_documents = []
for document in documents:
    tokenized_documents.append(word_tokenize(document))
    
small_tokens = []
for tokenized_document in tokenized_documents:
    for token in tokenized_document:
        if(len(token) <= 2):
            small_tokens.append(token)
small_tokens = list(set(small_tokens))
#print(small_tokens)

#Now it's time to remove punctuation marks
bad_tokens = [',', '?', '”', ')', '’', ';', '“', '.', '$', '(', ':', '!']

tokenized_documents = [[token for token in tokenized_document if token not in bad_tokens] 
                       for tokenized_document in tokenized_documents]

#Now before going for stop words, we see if any punctuation mark is left in any token

non_clear_tokens = []
for tokenized_document in tokenized_documents:
    for token in tokenized_document:
        non_clear_tokens.extend(re.findall(r'\w+[\'’:""();.?—-]\w*', token))
#print(non_clear_tokens)

#We see that we still have . and ? and — and - and – at the end or in the middle of the tokens
#So now we clear . and ? and — and - and – from end of tokens and for the middle ones we
#devide them into two tokens
tokenized_documents = [[token.rstrip('.') for token in tokenized_document] 
                       for tokenized_document in tokenized_documents]
tokenized_documents = [[token.rstrip('?') for token in tokenized_document] 
                       for tokenized_document in tokenized_documents]
tokenized_documents = [[token.rstrip('—') for token in tokenized_document] 
                       for tokenized_document in tokenized_documents]
tokenized_documents = [[token.rstrip('-') for token in tokenized_document] 
                       for tokenized_document in tokenized_documents]
tokenized_documents = [[token.rstrip('–') for token in tokenized_document] 
                       for tokenized_document in tokenized_documents]

new_tokenized_documents = []
for tokenized_document in tokenized_documents:
    new_tokenized_document = []
    for token in tokenized_document:
        if '.' in token:
            if(token.split('.')[0][:-1] > '9' or token.split('.')[-1][0] > '9'):
                new_tokenized_document.extend(token.split('.'))
            else:
                new_tokenized_document.append(token)
        elif '?' in token:
            new_tokenized_document.extend(token.split('?'))
        elif '-' in token:
            new_tokenized_document.extend(token.split('-'))
        elif '—' in token:
            new_tokenized_document.extend(token.split('—'))
        elif '–' in token:
            new_tokenized_document.extend(token.split('–'))
        else:
            new_tokenized_document.append(token)
    new_tokenized_documents.append(new_tokenized_document)

tokenized_documents = new_tokenized_documents

#Now that we donn't have non-meaning tokens (and punctuations) it's time for stop words
#Here we sort our tokens and count how many of each do we have
#And we try to find top 10 most repeated ones and we assume that they are all stop words

all_tokens = [token for tokenized_document in tokenized_documents for token in tokenized_document]

all_tokens.sort()

counted_tokens = []

ls = -1
for i in range(0, len(all_tokens)):
    if ((i + 1) == len(all_tokens)) or (all_tokens[i+1] != all_tokens[i]):
        counted_tokens.append([i - ls, all_tokens[i]])
        ls = i
        
counted_tokens.sort()
        
#print(counted_tokens[-10:])

#Now that we have our stop words too, it's time to remove them too and go for building our inverted index
#We also make a backup of our all tokens for our misspeling part
all_tokens_MS = list(set([all_tokens[i] for i in range(len(all_tokens))]))

stop_words = [a[1] for a in counted_tokens[-10:]]
tokenized_documents = [[token for token in tokenized_document if token not in stop_words] 
                       for tokenized_document in tokenized_documents]

#Now befor going any further we check our tokens for last time
#print(tokenized_documents)

#To build our inverted index, at first we need to unique our tokens
all_tokens = [token for tokenized_document in tokenized_documents for token in tokenized_document]
all_tokens = list(set(all_tokens))
#print(len(all_tokens))

#Now it's time to build a hash function to map our tokens to numbers between 0 to 1291
#Our hash function hash each token into it's equal number in base 256 moduled by 1313(our tokens number)
#And then it assignes the token to that number except for the times we encounter duplicate numbers
#In that case we go further untill we find an empty cell to assign our token to it
#And for finding hash number of a token(after assigning) we use this function again and if it finds the token it returns it.
all_tokens = len(all_tokens) * ['#']
def hash_func(token):
    B = 256; M = len(all_tokens); cur = 0
    for i in range(len(token)):
        cur = ((cur * B) + ord(token[i])) % M
    while(all_tokens[cur] != token and all_tokens[cur] != '#'):
        cur = (cur + 1) % M
    if(all_tokens[cur] == '#'):
        all_tokens[cur] = token
    return cur

#Now it's time to build our posting list
#We have a posting list which is a linked list of a index list for each document
#And each index list is a linked list of indexes of any token happening in a document
class Node:
    def __init__(self, order):
        self.order = order
        self.next = None

class Document_Node:
    def __init__(self, document_id):
        self.document_id = document_id
        self.head = None
        self.tail = None
        self.next = None
        
    def insert(self, order):
        node = Node(order)
        if(self.head == None):
            self.head = node
            self.tail = node
        else:
            self.tail.next = node
            self.tail = node
    
    def get_orders(self):
        orders = []
        current = self.head
        while current:
            orders.append(current.order)
            current = current.next
        return orders

class PostingList:
    def __init__(self):
        self.head = None
        self.tail = None

    def insert(self, document_id, order):
        if self.head == None:
            self.head = Document_Node(document_id)
            self.tail = self.head
            self.head.insert(order)
        else:
            current = self.head
            while (current != None) and (current.document_id != document_id):
                current = current.next
            if(current != None):
                current.insert(order)
            else:
                self.tail.next = Document_Node(document_id)
                self.tail = self.tail.next
                self.tail.insert(order)

    def get_documents(self):
        documents = []
        current = self.head
        while current:
            documents.append(current)
            current = current.next
        return documents

#Now it's time to fill our posting lists
posting_lists = [PostingList() for i in range(len(all_tokens))]

for i in range(len(tokenized_documents)):
    for j in range(len(tokenized_documents[i])):
        token = tokenized_documents[i][j]
        posting_lists[hash_func(token)].insert(i, j)

#Here we check if our posting lists are filled correctly
#for token in all_tokens:
#    print(token)
#    documents = posting_lists[hash_func(token)].get_documents()
#    for document in documents:
#        print(document.document_id)
#        print(document.get_orders())

#Now before handling queries, we should find misspeling candidates
#And also we should find all wildcart candidates and then we should
#calculate answer for each possible query candidate

#First we start with finding all misspeling candidates
#we call tokens like A candidate for given query word B
#If A's Levenshtein distance to B is minimum for all of our selected tokens
#we select tokens that start with the same character of B (our heuristic)

def calculate_dis(inp1, inp2):
    n = len(inp1); m = len(inp2)
    dis = [[n + m for i in range(m + 1)] for j in range(n + 1)]
    dis[0][0] = 0
    for i in range(n + 1):
        for j in range(m + 1):
            if(i > 0 and j > 0):
                dis[i][j] = min(dis[i][j], dis[i-1][j-1] + (inp1[i - 1] != inp2[j - 1]))
            if(i > 0):
                dis[i][j] = min(dis[i][j], dis[i-1][j] + 1)
            if(j > 0):
                dis[i][j] = min(dis[i][j], dis[i][j-1] + 1)
    return dis[n][m]

def find_all_closest(inp):
    selected_tokens = []
    mn = -1
    for token in all_tokens_MS:
        if(token[0] != inp[0]):
            continue
        cur = calculate_dis(token, inp)
        if mn == -1 or cur < mn:
            mn = cur
            selected_tokens = [token]
        elif mn == cur:
            selected_tokens.append(token)
    return selected_tokens

#Now that we have all closest misspelling candidates,
#its time to find all wildcard candidates, for this
#we use k-gram(k=2) algorithm and after finding first-step-candidates
#we filter the false possitives and return all remaining tokens

#first we build a posting list for our 2-grams
class KGNode:
    def __init__(self, token):
        self.token = token
        self.next = None

class KGList:
    def __init__(self):
        self.head = None
        self.tail = None
        
    def insert(self, token):
        node = KGNode(token)
        if(self.head == None):
            self.head = node
            self.tail = node
        else:
            self.tail.next = node
            self.tail = node
    
    def get_tokens(self):
        tokens = []
        current = self.head
        while current:
            tokens.append(current.token)
            current = current.next
        return tokens

#Now we make 2 2d list to save k-gram posting lists of each 2-gram
#and then we fill it with all our tokens
KGLists = [[KGList() for j in range(256)] for i in range(256)]

for token in all_tokens:
    new_token = '#' + token + '#'
    for i in range(len(new_token) - 1):
        KGLists[ord(new_token[i])][ord(new_token[i + 1])].insert(token)

#Now it's time to develop our last function which gets all tokens
#of all our 2-grams and find their intersection and removes it's false possitives

def get_intersection(arr1, arr2):
    arr2.sort()
    intersection = []
    
    pnt1 = 0
    for pnt2 in range(len(arr2)):
        while(pnt1 < len(arr1) and arr1[pnt1] < arr2[pnt2]):
            pnt1 += 1
        if(pnt1 < len(arr1) and arr1[pnt1] == arr2[pnt2]):
            intersection.append(arr1[pnt1])
    return intersection   

def get_all_wildcards(inp):
    new_inp = '#' + inp + '#'
    all_candidates = [token for token in all_tokens]; all_candidates.sort()
    for i in range(len(new_inp) - 1):
        if(new_inp[i] == '*' or new_inp[i + 1] == '*'):
            continue
        all_candidates = get_intersection(all_candidates, KGLists[ord(new_inp[i])][ord(new_inp[i + 1])].get_tokens())

    inp = inp.split('*')
    
    new_all_candidates = []
    for token in all_candidates:
        flag = True
        flag &= token[:len(inp[0])] == inp[0]
        flag &= (len(inp[-1]) == 0 or token[-len(inp[-1]):] == inp[-1])
        if(flag):
            new_all_candidates.append(token)
    all_candidates = new_all_candidates        
    
    if(len(inp) == 3):
        new_all_candidates = []
        for token in all_candidates:
            if(len(inp[-1]) != 0 and inp[1] in token[len(inp[0]):-len(inp[-1])]):
                new_all_candidates.append(token)
            elif(len(inp[-1]) == 0 and inp[1] in token[len(inp[0]):]):
                new_all_candidates.append(token)
        all_candidates = new_all_candidates
    
    return all_candidates

#Now that we can get all misspelling and wildcard candidates,
#It's time to handle our queries, for handling a query we find all candidate
#queries of the given query ans we answer each individually and also at last we answer their total union

#First we develop our And_handler in which we find document ids of each token and with two-pointer algorithm we
#find the intersection of our two lists

#Second we develop our Or_handler in which it adds second token's documents
#into the first one and uniques them and returns

#Third we develop our Not_handler in which in works much like our And_handler again but this time it's removing
#documents our token's is in from all documents list

#Fourth and at last, we develop our Near_handler in which we first find each tokens documents and 
#then we find intersection of those two lists with two-pointer algorithm and for each intersection we find now we get index lists of both
#of them and we use two-pointer algorithm again and we check for each index in first list if it's nearest smaller token in
#second list is at most k places further.

def And_handler(inp):
    targets = []
    if (inp[0] not in all_tokens) or (inp[2] not in all_tokens):
        return targets

    documents1 = posting_lists[hash_func(inp[0])].get_documents()
    documents1 = [document.document_id for document in documents1]

    documents2 = posting_lists[hash_func(inp[2])].get_documents()
    documents2 = [document.document_id for document in documents2]

    pnt1 = 0; pnt2 = 0
    while (pnt1 != len(documents1)) and (pnt2 != len(documents2)):
        if(documents1[pnt1] < documents2[pnt2]):
            pnt1 += 1
        elif(documents2[pnt2] < documents1[pnt1]):
            pnt2 += 1
        else:
            targets.append(documents1[pnt1])
            pnt1 += 1; pnt2 += 1

    return targets

def Or_handler(inp):
    targets = []
    if (inp[0] not in all_tokens) and (inp[2] not in all_tokens):
        return targets

    documents1 = []
    if(inp[0] in all_tokens):
        documents1 = posting_lists[hash_func(inp[0])].get_documents()
        documents1 = [document.document_id for document in documents1]

    documents2 = []
    if(inp[2] in all_tokens):
        documents2 = posting_lists[hash_func(inp[2])].get_documents()
        documents2 = [document.document_id for document in documents2]

    targets = documents1
    targets.extend(documents2)
    targets = list(set(targets))
    return targets

def Not_handler(inp):
    not_documents = []
    if(inp[1] in all_tokens):
        not_documents = posting_lists[hash_func(inp[1])].get_documents()
        not_documents = [document.document_id for document in not_documents]

    targets = []
    pnt2 = 0
    for pnt1 in range(len(tokenized_documens)):
        while(pnt2 < len(not_documents) and not_documents[pnt2] < pnt1):
            pnt2 += 1
        if not (pnt2 < len(not_documents) and not_documents[pnt2] == pnt1):
            targets.append(pnt1)

    return targets

def Near_handler(inp):
    dis = int(inp[1][5:]) + 1

    targets = []
    if (inp[0] not in all_tokens) or (inp[2] not in all_tokens):
        return targets

    documents1 = posting_lists[hash_func(inp[0])].get_documents()
    documents2 = posting_lists[hash_func(inp[2])].get_documents()

    pnt1 = 0; pnt2 = 0
    while (pnt1 != len(documents1)) and (pnt2 != len(documents2)):
        if(documents1[pnt1].document_id < documents2[pnt2].document_id):
            pnt1 += 1
        elif(documents2[pnt2].document_id < documents1[pnt1].document_id):
            pnt2 += 1
        else:
            orders1 = documents1[pnt1].get_orders()
            orders2 = documents2[pnt2].get_orders()

            order_pnt2 = 0
            for order_pnt1 in range(len(orders1)):
                while(order_pnt2 < len(orders2) and orders2[order_pnt2] < orders1[order_pnt1]):
                    order_pnt2 += 1
                if (order_pnt2 < len(orders2) and (orders2[order_pnt2] - orders1[order_pnt1]) <= dis):
                    targets.append(documents1[pnt1].document_id)
                    break
                if (order_pnt2 > 0 and (orders1[order_pnt1] - orders2[order_pnt2 - 1]) <= dis):
                    targets.append(documents1[pnt1].document_id)
                    break

            pnt1 += 1; pnt2 += 1
    return targets

def Query_handler(inp):
    inp = inp.split(' ')
    candidate_inp = []
    for i in range(len(inp)):
        if i == 1 and (inp[i] == "AND" or inp[i] == "OR" or inp[i] == "NOT" or inp[i][:4] == "NEAR"):
            candidate_inp.append([inp[i]])
            continue
        inp[i] = inp[i].lower()
        if '*' in inp[i]:
            candidate_inp.append(get_all_wildcards(inp[i]))
        else:
            candidate_inp.append(find_all_closest(inp[i]))
            
    if(len(candidate_inp) == 3 and candidate_inp[1] == ["AND"]):
        print("ALL possible queries are:")
        all_answers = []
        for inp0 in candidate_inp[0]:
            for inp2 in candidate_inp[2]:
                print(inp0, "AND", inp2)
                cur = And_handler([inp0, "AND", inp2])
                print(cur)
                all_answers.extend(cur)
                
        all_answers = list(set(all_answers))
        print("All answers in total are:")
        print(all_answers)
        
    elif(len(candidate_inp) == 3 and candidate_inp[1] == ["OR"]):
        print("ALL possible queries are:")
        all_answers = []
        for inp0 in candidate_inp[0]:
            for inp2 in candidate_inp[2]:
                print(inp0, "OR", inp2)
                cur = Or_handler([inp0, "OR", inp2])
                print(cur)
                all_answers.extend(cur)
                
        all_answers = list(set(all_answers))
        print("All answers in total are:")
        print(all_answers)
        
    elif(len(candidate_inp) == 2 and candidate_inp[0] == ["NOT"]):
        print("ALL possible queries are:")
        all_answers = []
        for inp0 in candidate_inp[0]:
            print("NOT", inp0)
            cur = Not_handler(["NOT", inp0])
            print(cur)
            all_answers.extend(cur)
        all_answers = list(set(all_answers))
        print("All answers in total are:")
        print(all_answers)
        
    elif(len(candidate_inp) == 3 and candidate_inp[1][0][:4] == "NEAR"):
        print("ALL possible queries are:")
        all_answers = []
        for inp0 in candidate_inp[0]:
            for inp2 in candidate_inp[2]:
                print(inp0, candidate_inp[1][0], inp2)
                cur = Near_handler([inp0, candidate_inp[1][0], inp2])
                print(cur)
                all_answers.extend(cur)
                
        all_answers = list(set(all_answers))
        print("All answers in total are:")
        print(all_answers)

    else:
        print("Invalid input")

#And here we answer queries with our query handler funtion which splits each query and base on it's form
#it calls the propriate function and also you can check misspelling and wildcard part here too
        
while(True):
    print("Please type you requests id")
    print("1. Giving query for IR system")
    print("2. Checking misspelling part")
    print("3. Checking wildcards part")
    print("4. EXIT")
    inp = int(input())
    
    if inp == 1:
        inp = input("Please type your query")
        Query_handler(inp)
        
    elif inp == 2:
        inp = input("Please type you words")
        inp = inp.lower().split(' ')
        for word in inp:
            print(find_all_closest(word))
            
    elif inp == 3:
        inp = input("Please type you words")
        inp = inp.lower().split(' ')
        for word in inp:
            print(get_all_wildcards(word))

    elif inp == 4:
        break
    else:
        print("Invalid input")
    print("####################")

