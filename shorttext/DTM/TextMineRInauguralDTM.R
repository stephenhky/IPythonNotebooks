
library(dplyr)
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

get.doc.tokens<- function(dtm, docid) 
  dtm[docid, ] %>% as.data.frame() %>% rename(count=".") %>% 
  mutate(token=row.names(.)) %>% arrange(-count)

get.token.docs<- function(dtm, token)
  dtm[, token] %>% as.data.frame() %>% rename(count=".") %>% mutate(token=row.names(.)) %>% arrange(-count) 

get.total.freq<- function(dtm, token) dtm[, token] %>% sum

dtm %>% get.doc.tokens('2009-Obama') %>% head(10)

dtm %>% get.token.docs(wordStem('change')) %>% head(10)

dtm %>% get.total.freq(wordStem('change'))
