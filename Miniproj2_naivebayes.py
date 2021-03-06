import sys
import os.path
import numpy as np
from collections import Counter
from collections import defaultdict

import util

USAGE = "%s <test data folder> <spam folder> <ham folder>"

def get_counts(file_list):
    """
    Computes counts for each word that occurs in the files in file_list.

    Inputs
    ------
    file_list : a list of filenames, suitable for use with open() or 
                util.get_words_in_file()

    Output
    ------
    A dict whose keys are words, and whose values are the number of files the
    key occurred in.
    """
    ### TODO: Comment out the following line and write your code here
    word_counter = Counter()
    for i in range(len(file_list)):
        words_in_file = list(set(util.get_words_in_file(file_list[i])))
        for word in words_in_file:
            word_counter[word] += 1
    
    return(word_counter)
    #raise NotImplementedError

def get_log_probabilities(file_list):
    """
    Computes log-frequencies for each word that occurs in the files in 
    file_list.

    Input
    -----
    file_list : a list of filenames, suitable for use with open() or 
                util.get_words_in_file()

    Output
    ------
    A dict whose keys are words, and whose values are the log of the smoothed
    estimate of the fraction of files the key occurred in.

    Hint
    ----
    The data structure util.DefaultDict will be useful to you here, as will the
    get_counts() helper above.
    """
    N = len(file_list)
    log_probs = defaultdict(lambda:np.log(1/(N+2)))
    word_counts = get_counts(file_list)
    words_list = word_counts.keys()
    for word in words_list:
        log_probs[word] = np.log(word_counts[word]/N)
    
    return log_probs
    
    ### TODO: Comment out the following line and write your code here
    #raise NotImplementedError


def learn_distributions(file_lists_by_category):
    """
    Input
    -----
    A two-element list. The first element is a list of spam files, 
    and the second element is a list of ham (non-spam) files.

    Output
    ------
    (log_probabilities_by_category, log_prior)

    log_probabilities_by_category : A list whose first element is a smoothed
                                    estimate for log P(y=w_j|c=spam) (as a dict,
                                    just as in get_log_probabilities above), and
                                    whose second element is the same for c=ham.

    log_prior_by_category : A list of estimates for the log-probabilities for
                            each class:
                            [est. for log P(c=spam), est. for log P(c=ham)]
    """
    spam_files = file_lists_by_category[0]
    ham_files = file_lists_by_category[1]
    No_spam_files = len(spam_files)
    No_ham_files = len(ham_files)
    log_prior_spam = np.log(No_spam_files/(No_spam_files+No_ham_files))
    log_prior_ham = np.log(No_ham_files/(No_spam_files+No_ham_files))
    log_prob_spam = get_log_probabilities(spam_files)
    log_prob_ham = get_log_probabilities(ham_files)
    
    log_probabilities_by_category = [log_prob_spam,log_prob_ham]
    log_prior_by_category = [log_prior_spam,log_prior_ham]
    return (log_probabilities_by_category,log_prior_by_category)
    
    
    ### TODO: Comment out the following line and write your code here
    #raise NotImplementedError

def classify_email(email_filename,
                   log_probabilities_by_category,
                   log_prior_by_category):
    """
    Uses Naive Bayes classification to classify the email in the given file.

    Inputs
    ------
    email_filename : name of the file containing the email to be classified

    log_probabilities_by_category : See output of learn_distributions

    log_prior_by_category : See output of learn_distributions

    Output
    ------
    One of the labels in names.
    """
    words_in_email = list(set(util.get_words_in_file(email_filename)))
    log_prob_spam = log_probabilities_by_category[0]
    log_prob_ham = log_probabilities_by_category[1]
    all_words = set(list(list(log_prob_spam.keys())+list(log_prob_ham.keys())))
    P_spam = log_prior_by_category[0]+ np.sum([log_prob_spam[word] for word in words_in_email])+\
                np.sum([np.log(1 - np.exp(log_prob_spam[word])) for word  in list(set(all_words) - set(words_in_email))])
    
    P_ham = log_prior_by_category[1]+ np.sum([log_prob_ham[word] for word in words_in_email])+\
                 np.sum([np.log(1 - np.exp(log_prob_ham[word])) for word  in list(set(all_words) - set(words_in_email))])
    if (P_spam>= P_ham):
        return('spam')
    else:
        return('ham')
    ### TODO: Comment out the following line and write your code here
    #return 'spam'

def classify_emails(spam_files, ham_files, test_files):
    # DO NOT MODIFY -- used by the autograder
    log_probabilities_by_category, log_prior = \
        learn_distributions([spam_files, ham_files])
    estimated_labels = []
    for test_file in test_files:
        estimated_label = \
            classify_email(test_file, log_probabilities_by_category, log_prior)
        estimated_labels.append(estimated_label)
    return estimated_labels

def main():
    ### Read arguments
    if len(sys.argv) != 4:
        print(USAGE % sys.argv[0])
    testing_folder = sys.argv[1]
    (spam_folder, ham_folder) = sys.argv[2:4]

    ### Learn the distributions
    file_lists = []
    for folder in (spam_folder, ham_folder):
        file_lists.append(util.get_files_in_folder(folder))
    (log_probabilities_by_category, log_priors_by_category) = \
            learn_distributions(file_lists)

    # Here, columns and rows are indexed by 0 = 'spam' and 1 = 'ham'
    # rows correspond to true label, columns correspond to guessed label
    performance_measures = np.zeros([2,2])

    ### Classify and measure performance
    print("Testing...")
    idx = 1
    for filename in (util.get_files_in_folder(testing_folder)):
        print(idx)
        print(filename)
        idx += 1
        ## Classify
        label = classify_email(filename,
                               log_probabilities_by_category,
                               log_priors_by_category)
        ## Measure performance
        # Use the filename to determine the true label
        base = os.path.basename(filename)
        true_index = ('ham' in base)
        guessed_index = (label == 'ham')
        performance_measures[true_index, guessed_index] += 1


        # Uncomment this line to see which files your classifier
        # gets right/wrong:
        print("%s : %s" %(label, filename))

    template="You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
    # Correct counts are on the diagonal
    correct = np.diag(performance_measures)
    # totals are obtained by summing across guessed labels
    totals = np.sum(performance_measures, 1)
    print(template % (correct[0],
                      totals[0],
                      correct[1],
                      totals[1]))

if __name__ == '__main__':
    main()
