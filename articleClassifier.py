#INCLUDE
# imports ADD ALL the NECESSARY imports
import numpy as np
import operator
import string 
import nltk
import re
from nltk.tokenize import sent_tokenize, word_tokenize 
from nltk.corpus import stopwords
stop_words = stopwords.words('french')

"""
stop_words = ['au', 'aux', 'avec', 'ce', 'ces', 'dans', 'de', 'des', 'du', 'elle', 'en', 'et', 'eux', 'il', 'ils', 'je', 'la', 'le', 'les', 'leur', 'lui', 'ma', 'mais', 'me', 'même', 'mes', 'moi', 'mon', 'ne', 'nos', 'notre', 'nous', 'on', 'ou', 'par', 'pas', 'pour', 'qu', 'que', 'qui', 'sa', 'se', 'ses', 'son', 'sur', 'ta', 'te', 'tes', 'toi', 'ton', 'tu', 'un', 'une', 'vos', 'votre', 'vous', 'c', 'd', 'j', 'l', 'à', 'm', 'n', 's', 't', 'y', 'été', 'étée', 'étées', 'étés', 'étant', 'étante', 'étants', 'étantes', 'suis', 'es', 'est', 'sommes', 'êtes', 'sont', 'serai', 'seras', 'sera', 'serons', 'serez', 'seront', 'serais', 'serait', 'serions', 'seriez', 'seraient', 'étais', 'était', 'étions', 'étiez', 'étaient', 'fus', 'fut', 'fûmes', 'fûtes', 'furent', 'sois', 'soit', 'soyons', 'soyez', 'soient', 'fusse', 'fusses', 'fût', 'fussions', 'fussiez', 'fussent', 'ayant', 'ayante', 'ayantes', 'ayants', 'eu', 'eue', 'eues', 'eus', 'ai', 'as', 'avons', 'avez', 'ont', 'aurai', 'auras', 'aura', 'aurons', 'aurez', 'auront', 'aurais', 'aurait', 'aurions', 'auriez', 'auraient', 'avais', 'avait', 'avions', 'aviez', 'avaient', 'eut', 'eûmes', 'eûtes', 'eurent', 'aie', 'aies', 'ait', 'ayons', 'ayez', 'aient', 'eusse', 'eusses', 'eût', 'eussions', 'eussiez', 'eussent']
"""

class ArticleClassifier:

    def __init__(self,n_sig_words=3,min_score=0.5,t=250):
        # HyperParameters
        self.n_sig_words=n_sig_words
        self.min_score = min_score
        self.t = t
        self.epsilon = 0.0001
        self.rounding = 3 # Number of significant numbers
        self.related_words = {}
        self.article_label_set = list()
        # Predition attributes
        self.pred_eval = list() # Tag each prediction 1:correct, 0:wrong for each article
        self.pred_labels = list() # Siren predicted for each article
        self.article_eval = list() # Tag each label if 1:predicted, 0:not predicted for each article
        self.article_labels = list() # Siren labels for each article
        self.pred_labels_flat = list() #list all predicted sirens flattened
        self.article_labels_flat = list() # list of all siren labels flattened
        # Evaluation attributes
        self.score1 = 0       
        self.score2 = 0
        self.score3 = 0       
        self.score4 = 0         
        self.avg_n_pred = 0       
        self.avg_n_labels = 0        
        self.most_commun_label = 0
        # Company evaluation
        self.company_positive = list()
        self.company_accuracy_list = list()
        self.company_precision_list = list()
        self.company_recall_list = list()
        self.company_F1score_list = list()
        # Article evaluation
        self.alpha_eval_list = list()
        self.article_recall_list = list()
        self.article_precision_list = list()
        self.alpha = 1   # penalizes errors if >1 hides errors if <1
        self.beta = 0.25 # weight for the missed labels (False Negative)
        self.gamma = 1   # weight for the wrongly predicted (False positives)
        
    #################################################################
    # Adds scored relevant words to the model
    #Input  : relevant word dictionary
    #Output : Text removing all punctuation and lowercased
    #################################################################
    def fit(self, relevant_word_dict):
        self.related_words = relevant_word_dict
        self.article_label_set = list(relevant_word_dict.keys())
    
    
    
    #################### PREDICTION FUNCTIONS #################################################################
    
    #################################################################
    # CLEANING PLAIN TEXT (use on un tockenize, uncleaned text)
    # Input  : Plain text - String
    # Output : Text removing all punctuation and lowercased
    #################################################################
    def clean_plain_text(self,text):
        text = text.lower() # lower
        text = text.translate(str.maketrans("","", string.punctuation)) # removing punctuation
        text = re.sub(r'»|«|–|…', '', text)  # suprime guillmets 
        text = re.sub(r'è|é|ê|ë|ē|ė|ę', 'e', text)  # suprime accents sur le e
        text = re.sub(r'à|á|â|ä|æ|ã|å|ā', 'a', text)  # suprime accents sur le a
        text = re.sub(r'\s+', ' ', text) # remove everything else
        text = word_tokenize(text)
        cleaned_text = list()
        for word in text:
            if len(word)>1:
                if word not in stop_words:
                    cleaned_text.append(word)
        text = cleaned_text
        return text

    
    
    #################################################################
    # Gives a companies "related score" wrt an article (using it's significant words)
    #INPUT :plain_text- String/ word_list - list of significant words
    #OUTPUT: Score the chances the company is related to the article
    #################################################################
    def company_relevance_score(self,plain_text,sig_words_list): 
        sig_words = np.array(sig_words_list)[:,0]
        sig_words_score = np.array(sig_words_list)[:,1].astype(np.float) #CHANGED
        sig_words_score = sig_words_score/(np.sum(sig_words_score) +self.epsilon)# normalizing all of the sig_words scores
        sum_exp = np.sum([np.exp(float(score)) for score in sig_words_score]) # denominator for computing the soft max
        n_words = len(plain_text)

        words_in_text = 0
        for i in range(len(sig_words_list)):
            word_soft_max = np.exp(float(sig_words_score[i]))/(sum_exp +self.epsilon)
            words_in_text += word_soft_max*plain_text.count(sig_words[i])
        
        return words_in_text/(n_words+self.epsilon) # relevance score for the company
    
    
    
    #################################################################
    # For an Article, gives the "related scores"(likeness of being a label) for all companies
    #INPUT :plain_text- String/company related words - dict/ params
    #OUTPUT: dict of companies and their "related scores"
    #################################################################
    def text_label_scores(self,plain_text, criterion = "T"):
        label_dict = {}
        label_dict_res = {}
        for siren in self.related_words.keys():
            sig_words_list = np.array(self.related_words[siren])[:self.n_sig_words] # Build significant word list (with no scores)
            score = self.company_relevance_score(plain_text, sig_words_list)
            score = 1 - 1/(1 + self.t*score) # smooth relevant scores
            label_dict[siren]= score

        label_dict = {k: v for k, v in sorted(label_dict.items(), key=lambda item: -item[1])} # sort all companies wrt score

        for label in label_dict.keys():
            if label_dict[label]>=self.min_score:
                label_dict_res[label] = label_dict[label]
        if (criterion.lower() == 't'):
            if label_dict_res == {}:
                label_dict_res[list(label_dict.keys())[0]] = label_dict[list(label_dict.keys())[0]]

        return label_dict_res #relevance score for each company
    
    
    
    #################################################################
    # For an Article, predicts the labels (sirens)
    #INPUT : plain_text- String/company related words - dict/ params
    #OUTPUT: dict of companies and their "related scores"
    #################################################################
    def label_text(self,plain_text,criterion = "T"):
        label_dict = self.text_label_scores(plain_text,criterion)
        sirens = list(label_dict.keys())
        return sirens # No limitation of the number of labels 
    
    
    #################################################################
    # Predict le labels of a given corpus wrt. the given hyper parameters
    #INPUT : Corpus, hyper parameters
    #OUTPUT: Predicted companies for each article
    #################################################################
    def predict(self,corpus,max_n_pred = None, criterion = "T"):
        #for document in tqdm(corpus):
        for document in corpus:
            plain_text = document["corpus"]
            
            #self.pred_labels
            pred_sirens = self.label_text(plain_text, criterion)[:max_n_pred] # predict labels
            self.pred_labels.append(pred_sirens)
            #self.pred_labels_flat
            self.pred_labels_flat += pred_sirens

            #self.article_labels
            true_sirens =document["siren"]
            self.article_labels.append(true_sirens)
            #self.article_labels_flat
            self.article_labels_flat +=true_sirens

            #self.pred_eval 
            is_labeled = [0]*len(pred_sirens)
            for i in range(len(pred_sirens)):  # For each predicted company
                for label in true_sirens: # For each labeled company
                    if pred_sirens[i]==label:  # Tag if it is a good or bad predictions
                        is_labeled[i]=1
            self.pred_eval.append(is_labeled) 

            #self.article_eval
            is_predicted = [0]*len(true_sirens)
            for i in range(len(true_sirens)):  # For each label list
                for pred in pred_sirens:       # For each prediction on the articel
                    if true_sirens[i]==pred:    # Tag the labels that have been predicted
                        is_predicted[i]=1
            self.article_eval.append(is_predicted)
        return self.pred_labels
    
    #################### MODEL EVALUATION FUNCTIONS ###########################################################
    
    #################################################################
    # Generate scores for evaluating the model
    # INPUT: 
    # OUTPUT: 
    #################################################################
    def evaluate(self):
        ########## How many times (at least) one of the companies is predicted ##########
        acc1 = list()
        for preds in self.pred_eval:
            acc1.append(any(preds))
        self.score1 = round(np.sum(acc1)/(len(self.pred_eval)+self.epsilon),self.rounding)

        ########## How many times ALL the labels are present in the prediction. ##########
        acc2 = list()
        for labels in self.article_eval:
            acc2.append(labels.count(1)== len (labels)) 
        self.score2 = round(np.sum(acc2)/(len(self.pred_eval)+self.epsilon),self.rounding)

        ########## How many times ALL labels are predicted in the FIRST predictions. ##########
        acc3 = list()
        for i in range(len(self.pred_eval)):
            labels = self.article_eval[i]
            preds = self.pred_eval[i]
            acc3.append(preds[:len(labels)].count(1)== len(labels))
        self.score3 =  round(np.sum(acc3)/(len(self.pred_eval)+self.epsilon),self.rounding)

        ########## How many predictions are wrong wrt. how many are right (TRUE, FALSE) ##########
        true_pred = 0
        pred = 0
        for preds in self.pred_eval:
            true_pred += np.sum(preds)
            pred += len(preds)
        self.score4 = round(true_pred/(pred + self.epsilon),self.rounding)

        ########## Average number of predictions vs average number of labels ##########
        len_label = list()
        len_pred = list()
        for i in range(len(self.pred_eval)):
            len_label.append(len(self.article_labels[i]))
            len_pred.append(len(self.pred_eval[i]))
        self.avg_n_pred = round(np.mean(len_pred),self.rounding)
        self.avg_n_labels = round(np.mean(len_label),self.rounding)

        ########## Most commun labels predicted ##########
        count_pred = dict()
        for siren in self.article_labels_flat:
            if siren in count_pred.keys():
                count_pred[siren] +=1
            else:
                count_pred[siren] = 1
        key_max = list(filter(lambda t: t[1]==max(count_pred.values()), count_pred.items()))[0][0] 
        self.most_commun_label = [key_max,np.max(list(count_pred.values()))]
        
        ########## Precision & RECALL ########## per siren(Company)

        for siren in self.article_label_set: # For each company compute it's TP,FP,TN,FN
            true_pos = 0.0  # Siren IS a label and is predicted
            false_pos = 0.0 # Siren is NOT a label and is predicted (false prediction)
            true_neg = 0.0  # Siren is NOT a label and is not predicted (don't care)
            false_neg = 0.0 # Siren IS a label and is NOT predicted
            positive = 0.0  # Siren is label

            # true_pos, false_neg
            for i in range(len(self.article_labels)):
                for j in range(len(self.article_labels[i])): # for every label
                    if siren==self.article_labels[i][j]: # If company in the list of labels -> Check if was predicted
                        positive +=1
                        if self.article_eval[i][j]==1:
                            true_pos +=1
                        else:
                            false_neg +=1

            # false_pos
            for i in range(len(self.pred_labels)):
                for j in range(len(self.pred_labels[i])):
                    if siren==self.pred_labels[i][j]:  # If company in the list of predictions -> Check if was a label (correct prediction)
                        if self.pred_eval[i][j]==0: #ie. Is not a label
                            false_pos += 1 

            if siren in list(set(self.article_labels_flat)): # Add to stats only if the company was part of the labels to predict
                if positive ==0: # redundancy with previous if statement
                    precision = 0
                    recall =0
                    accuracy = 0
                    F1score = 0
                else:
                    accuracy = true_pos/(positive + self.epsilon)
                    precision = true_pos/(true_pos+false_pos + self.epsilon)
                    recall = true_pos/(true_pos+false_neg + self.epsilon)
                    F1score = 2*(precision*recall)/(precision+recall+ self.epsilon)

                self.company_positive.append(positive)
                self.company_accuracy_list.append(accuracy)
                self.company_precision_list.append(precision)
                self.company_recall_list.append(recall)
                self.company_F1score_list.append(F1score)

        ########## Precision & RECALL ########## per Article      
        #alpha_eval_list
        for i in range(len(self.article_labels)):
            alpha_eval = pow((1-((self.beta*self.article_eval[i].count(0) + self.gamma*self.pred_eval[i].count(0))/(len(set(self.pred_labels[i]+self.article_labels[i]))+ self.epsilon))),self.alpha) 
            self.alpha_eval_list.append(alpha_eval)
        #article_recall_list
        for label in self.article_eval:
            self.article_recall_list.append((label.count(1)+self.epsilon)/(len(label)+self.epsilon))
        #article_precision_list
        for pred in self.pred_eval:
             self.article_precision_list.append((pred.count(1)+self.epsilon)/(len(pred)+self.epsilon))
    
    def print_eval(self, verbose = 2):
        #verbose = 0,1
        if verbose == 0:
            print ("min_score :",self.min_score)
            print ("n_sig_words :",self.n_sig_words)
            print ("t :",self.t)
            print("Numbre of test articles :",len(self.pred_labels))
            print("AVG PRECISION per Company:",round(np.average(self.company_precision_list),self.rounding))
            print("AVG PRECISION per Article:",round(np.average(self.article_precision_list) ,self.rounding))
        
        elif verbose == 1:
            print ("min_score :",self.min_score)
            print ("n_sig_words :",self.n_sig_words)
            print ("t :",self.t)
            print("Number of test articles :",len(self.pred_labels))
            print("######################### For Each company #########################")
            print("AVG ACCURACY :",round(np.average(self.company_accuracy_list),self.rounding))
            print("AVG PRECISION:",round(np.average(self.company_precision_list),self.rounding))
            print("AVG RECALL   :",round(np.average(self.company_recall_list),self.rounding))
            print("AVG F1 score :",round(np.average(self.company_F1score_list),self.rounding))
            print()
            print("######################### For Each article #########################")
            print("AVG PRECISION:",round(np.average(self.article_precision_list) ,self.rounding))
            print("AVG RECALL   :",round(np.average(self.article_recall_list),self.rounding))
            print("AVG alpha eval:",round(np.average(self.alpha_eval_list),self.rounding))
   
        elif verbose == 2:
            print ("min_score :",self.min_score)
            print ("n_sig_words :",self.n_sig_words)
            print ("t :",self.t)
            print("Number of test articles :",len(self.pred_labels))
            print("Score 1:", self.score1,"(with at least ONE label predicted)")
            print("Score 2:", self.score2,"(with ALL labels predicted)")
            print("Score 3:", self.score3,"(with ALL labels predicted in the FIRST predictions)")
            print("Score 4:",self.score4,"(Number of correct predictions over total number of predictions overall)")
            print("Average number of predictions",self.avg_n_pred,"vs average number of labels :", self.avg_n_labels)
            print("The siren that is predicted the most is:",self.most_commun_label[0],"(",self.most_commun_label[1],"times)")
            print()
            print("######################### For Each company #########################")
            print("AVG ACCURACY :",round(np.average(self.company_accuracy_list),self.rounding),"True_pos/Pos -> average for each siren")
            print("AVG PRECISION:",round(np.average(self.company_precision_list),self.rounding),"True_pos/(True_Pos + False_Pos) -> average for each siren")
            print("AVG RECALL   :",round(np.average(self.company_recall_list),self.rounding),"True_pos/(True_Pos + False_Neg) -> average for each siren")
            print("AVG F1 score :",round(np.average(self.company_F1score_list),self.rounding),"combination of precision and recall -> average for each siren")
            print()
            print("######################### For Each article #########################")
            print("AVG PRECISION:",round(np.average(self.article_precision_list) ,self.rounding),"#correct_predictions/#predictions-> average for each article")
            print("AVG RECALL   :",round(np.average(self.article_recall_list),self.rounding),"#predicted_labels/#labels -> average for each article")
            print("AVG alpha eval:",round(np.average(self.alpha_eval_list),self.rounding),"prediction score of an article -> average for each article")
        