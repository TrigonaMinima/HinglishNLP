Transliteration Dataset
======

This readme explains the sources of the transliteration pairs used for the models. Publicly released datasets are linked to the sources and for synthetic dataset the whole process is documented along with the publications where these patterns were observed. Scripts for the scraped pairs are present in `datagen` directory.


# Publicly Available Datasets

1. [Xlit-Crowd: Hindi-English Transliteration Corpus](https://github.com/anoopkunchukuttan/crowd-indic-transliteration-data) [Small Size]

    > Mitesh M. Khapra, Ananthakrishnan Ramanathan, Anoop Kunchukuttan, Karthik Visweswariah, Pushpak Bhattacharyya. When Transliteration Met Crowdsourcing : An Empirical Study of Transliteration via Crowdsourcing using Efficient, Non-redundant and Fair Quality Control . Language and Resources and Evaluation Conference (LREC 2014). 2014.

    These pairs were obtained via crowdsourcing by asking workers to transliterate Hindi words into the Roman script. The tasks were done on Amazon Mechanical Turk and yielded a total of 14919 pairs.

2. [NeuralCharTransliteration](https://github.com/UtsabBarman/NeuralCharTransliteration) [Medium Size]

    Data is created from the scrapped Hindi songs lyrics. I have combined both `train_file.txt` and `test_file.txt`.

3. [Xlit-IITB-Par](http://www.cfilt.iitb.ac.in/iitb_parallel/) [Medium Size]

    > Anoop Kunchukuttan, Pratik Mehta, Pushpak Bhattacharyya. The IIT Bombay English-Hindi Parallel Corpus. Language Resources and Evaluation Conference. 2018.

    This is a corpus containing transliteration pairs for Hindi-English. These pairs were automatically mined from the IIT Bombay English-Hindi Parallel Corpus using the Moses Transliteration Module. The corpus contains 68,922 pairs.

4. [BrahmiNet Corpus: 110 language pairs](http://www.cfilt.iitb.ac.in/brahminet/static/download.html) [Small Size]

    > Anoop Kunchukuttan, Ratish Puduppully , Pushpak Bhattacharyya, Brahmi-Net: A transliteration and script conversion system for languages of the Indian subcontinent , Conference of the North American Chapter of the Association for Computational Linguistics â Human Language Technologies: System Demonstrations . 2015 [pdf](https://www.aclweb.org/anthology/N15-3017.pdf)

    The Brahmi-Net transliteration resources consist of parallel transliteration corpora for 110 language pairs, comprising 10 Indian languages and English. The transliteration corpus has been mined from the Indian Language Corpora Initiative (ILCI) corpus, containing tourism and health domains sentences. For our purposes we only picked

5. [Hindi word transliteration pairs 1](https://cse.iitkgp.ac.in/resgrp/cnerg/qa/fire13translit/index.html) [Small Size]

    > Kanika Gupta and Monojit Choudhury and Kalika Bali, Mining Hindi-English Transliteration Pairs from Online Hindi Lyrics, In Proceedings of the Eight International Conference on Language Resources and Evaluation (LREC '12), 23-25 May 2012, Istanbul, Turkey, pages 2459-2465. [pdf](http://www.lrec-conf.org/proceedings/lrec2012/pdf/365_Paper.pdf)

6. [Hindi word transliteration pairs 2](https://cse.iitkgp.ac.in/resgrp/cnerg/qa/fire13translit/index.html) [Medium Size]

    > Sowmya V.B., Monojit Choudhury, Kalika Bali, Tirthankar Dasgupta and Anupam Basu. "Resource Creation for Training and Testing of Transliteration Systems for Indian Languages", LREC 2010.

    This dataset contains sentences written in english annotated with the Hindi back-transliterated words. I have combined all the individual text files into a single text file. This file with serve two purposes - give us transliteration pairs and also usable (with a few modifications) in the final model where we'll be running our model at sentence level.


All the above 6 datasets are combined into a single file called - `en_hi_rel.csv`. We obtained 2,12,820 pairs in total.


# Data Creation

This section discusses the creation of dataset from Wikipedia. We downloaded the wikidata dumps having JSON entries on every entity and property. One useful property of these entries is that, labels, aliases and descriptions are given in multiple languages, all crowd-sourced of course. We specifically extracted the English and Hindi versions of each entity and property. This gave us a huge noisy parallel corpus of transliterations. Post this extraction, a lot of pre-processing steps were employed followed by a lot of manual sifting. At the end, we obtained a file with 2,17,424 pairs. Note that, the dataset contains a lot of duplicates as we wanted to keep the distribution of words as close to the usage as possible, although even with the duplicates it'll be far from the real-world usage. This will take care of the fact that frequency of some words is more in the real-world language usage and those words should definitely be correctly transliterated. I'll list down the steps in bullets, but a more detailed explanation can be found in this blog entry - []().


1. Download the Wikidata JSON dump from the [download page](https://dumps.wikimedia.org/wikidatawiki/entities/). I downloaded the `latest-all.json.bz2` (~38GB) created on `08-Oct-2019 11:29`;
2. Execute `datagen/wikidata2.py`, which will generate a `eng_hin_pairs.txt` having all the extracted eng-vs-hin pairs;
3. Divide the `eng_hin_pairs.txt` into 3 parts
    1. Instances where both eng and hin words have no spaces (`pairs1`);
    2. Instances where both eng and hin words have equal number of spaces (`pairs2`);
    3. All the remaining pairs (`pairs3`);
4. For `pairs2`, assuming the pairs they are aligned, split by space, zipped together and added all the single instances to the `pairs1` list;
5. For `pairs3`, took a cross-product of all the words from eng to hin and added all these single instances to the `pairs1` list;
6. Passed the `pairs1` list through a quick character mapping based transliteration function (this function gives a list of possible transliteration for a given `hin` word). All the pairs passing through this were valid transliterations
7. Then checked the remaining pairs with an edit distance of 0.9 or greater among the list of transliterations
8. Checked the remaining instances manually to get the final list:
    - Used regexes
    - Went one-by-one to eliminate the wrong pairs


I have expatiated about the steps and more in this blog post - []().


# Training Files Created

- Train 1 (`en_hi_rel.csv`): Combined Released Dataset
- Train 2 (`en_hi_wiki.csv`): Wiki + Manual Dataset
- Train 3 (`en_hi_com.csv`): Wiki + Manual + Combined Released Dataset
