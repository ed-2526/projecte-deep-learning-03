For this competition on information extraction we have manually labeled and checked a subset 
of the Barcelona Marriage Records written in old Catalan between 1617 and 1619.
This training dataset is composed of 968 Marriage Records.

For each record we provide line and word level segmented images, transcription, categories and persons.

The categories we can find are:
'name','surname','location','occupation','state'.

And the persons are:
'husband','wife','husbands_father','husbands_mother','wifes_father','wifes_mother','other_person'.

For those non-relevant words (e.g. conjunctions,prepositions,verbs, etc.) the category 
will be'other' and the person will be 'none'.

We also provide a sample of the output CSV file that will be used for validation.
For each record, a CSV file is generated. It only contains the relevant words, and follows this format:

transcription_word1,category_word1,person_word1
transcription_word1,category_word2,person_word2
transcription_word3,category_word3,person_word3

*For the track 1 (basic) there is no need to provide the person information in the CSV.

A list of all the characters that appear in the dataset can be found in chars.txt. 
There are 59 characters (plus the blank space for word separator if you are working at line level).
The character # represents crossed out words or characters.
All text files use utf-8 encoding to be able to represent the character 'ç'.