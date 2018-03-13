# AAAI18_983_Network_reproduction
*original code cloned from the author's repository: https://github.com/nday0739/bci_first*

**paper: https://arxiv.org/abs/1708.06578**


**Working on better code.**

This paper surprised my supervisor and me myself for its extremely high accuracy recognizing EEG signals. I managed to find the author's git repository but found some critical problems, which may suggest that the results in the paper are not convincing.

Instead of the traditional training, validating and testing pipeline, where we should consider the session-to-session and the subject-to-subject transfer and the way important BCI competitions and public datasets like the BBCI competitions took to strictly examine whether the algorithms being used are truly effective, the authors just shuffle all the 109 subjects' data and split them into training set and testing set with a 3:1 proportion. That's how the 98.3% accuracy was produced.

To save disk room I removed the raw data and useless code.
You can run the experiment yourself to see the results.


**update**

I've run these two experiments for several days and it's still running. The computation amount is much larger than I expected. However, the results are disappointing (the best accuracies are no more than 30%) and I try to figure out why.

The most important part is, I guess, is that the EEG data are poorly represented. How long does a motor imagery executation at least takes? The author just cropped the EEG data into a 3D cube: a 2-D map with 10 time-points. So can it contain some useful information about motor imagery? I doubt it.

I don't want to continue more experiments like the para-model. Another paper written by some Germans seems much more believable.