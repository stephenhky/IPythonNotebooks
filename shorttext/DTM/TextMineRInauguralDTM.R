
library(textmineR)
library(SnowballC)

usprez.df<- read.csv('inaugural.csv', stringsAsFactors = FALSE)
dtm<- CreateDtm(usprez.df$speech, 
                doc_names = usprez.df$yrprez, 
                ngram_window = c(1, 1),
                lower = TRUE,
                remove_punctuation = TRUE,
                remove_numbers = TRUE,
                stem_lemma_function = wordStem)
