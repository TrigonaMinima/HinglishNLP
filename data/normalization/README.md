Normalization Dataset
======

This readme explains the sources of the spelling error pairs used for the models. Publicly released datasets are linked to the sources and for synthetic dataset the whole process is documented along with the publications where these patterns were observed.


# Publicly Available Datasets

1. [EFC Corpus](https://corpus.mml.cam.ac.uk/efcamdat/) [Medium]

    J Geertzen, T Alexopoulou, and A Korhonen. 2012. Automatic Linguistic Annotation of Large Scale L2 Databases: The EF-Cambridge Open Language Database (EFCamDat). In Ryan T. Miller, editor, Selected Proceedings of the 2012 Second Language Research Forum. MA: Cascadilla Proceedings Project.

    Lisa Beinborn, Torsten Zesch, Iryna Gurevych. Predicting the Spelling Difficulty of Words for Language Learners. In:Proceedings of the 11th Workshop on Innovative Use of NLP for Building Educational Applications held in conjunction with NAACL 2016, p. to appear, 2016.

2. birkbeck (https://www.dcs.bbk.ac.uk/~ROGER/corpora.html) [Small]
3. holbrook (https://www.dcs.bbk.ac.uk/~ROGER/corpora.html) [Small]
4. aspell (https://www.dcs.bbk.ac.uk/~ROGER/corpora.html) [Small]
5. wikipedia (https://www.dcs.bbk.ac.uk/~ROGER/corpora.html) [Small]
6. [Facebook moe spelling dataset](https://github.com/facebookresearch/moe/tree/master/data) [Big]

> Note: Part of the spelling error pairs from data sources, 1 till 5 were combined together and manually sifted through. There were some invalid corrections which did not make any sense so I was trying to remove those. All these manually looked through pairs are in `train_manual_all.csv` file. Later I realised that finishing the whole list is going to take a lot of time, so I forgo the idea and put the rest of the pairs in `train_released_1.csv`.


# Synthetic Data Creation

## Typographic Errors

1. Insertion of single letters: "untill" for "until"
2. Omission of a single letter: "occuring" for "occurring"
3. Transposition of two consecutive letters: "freind" for "friend"
4. Substitution of one letter by another: "definate" for "definite"

## Cognitive Errors

1. Consonant doubling
2. Substitution of one letter by another: "definate" for "definite"
3. Apostrophe errors

> Note: Synthetic data and the script to create will be released once it's crystallized.


## Publications

- V.J. Cook (1997): L2 Users and English Spelling, Journalof Multilingual and Multicultural Development, 18:6, 474-488, DOI:10.1080/01434639708666335
- AKBAR SOLATI, AZIMAH SAZALIE & SALASIAH CHE LAH: Patterns of Spelling Errors in Language Learners' Language: An Investigation on Persian Learners of English
- Yves Bestgen and Sylviane Granger: Categorizing spelling errors to assess L2 writing
- Ibrahim Abdulrahman AHMED: Different types of spelling errors made by Kurdish EFL learners and their potential causes
- Kusuran, Amir: L2 English spelling error analysis - An investigation ofEnglish spelling errors made by Swedish senior high school students


## Training Files Created

- Train 1 (`train_0.csv`): Manual + Released Dataset
- Train 2 (`train_1.csv`): Manual + Released + Synthetic Dataset
- Train 3 (`train_2.csv`): Facebook MOE Dataset

> Note: Scripts to make these datasets will be released once they are crystallized.
