## Appendix

_Colourless Light_ is an attempt to visualise my own letter-number synaesthesia by tweaking the output of an RNN. I've noticed that lipograms - texts which avoid certain letters, the most famous of which is George Perec's novel _La Disparition_, translated into English as _A Void_ - look different in that area of my mind's eye in which words have colours. The letter E is a strong bright red, and most of the contribution to a word's synaesthetic hue comes from the vowels, so Perec's novel, getting its vowel colours from A (white), I (transparent), O (black) and U (deep purple), is darker and shifted towards the blue end of the spectrum.

I've been tinkering with lipogrammatic RNNs for a while now - for a recent experiment see [gravidum cor](https://bots.mikelynch.org/gravidum_cor/), a Mastodon bot which generates a simulation of Burton's Anatomy of Melancholy without E. What I wanted to do for this year's NaNoGenMo was make an RNN generate text with a different mixture of letters on different parts of the page. I had no idea how this would look to my synaesthetic sense - whether the invisible colours of the letters would translate when arranged spatially in this way.

At first I had the idea that one could take an image and map its RGB components onto the appropriate letters and make a "picture", but my early experiments with rendering two different constrained outputs into a checkerboard proved that patterns on a scale of ten or so letters weren't really detectable. For example, the following is a rendering of a checkerboard pattern where the squares are 20 characters wide and 10 lines tall, and the colours alternate between black and red. (Black has a lot of 'o' and red has a lot of 'e' and 's')

### Black and red checkerboard

```
 colours of our own surfaces essentially observations on others ese oneselfess, 
on the other own colours seen by semi-transparent forms, and seeses to see these
 operations of our own surfaces essays enough to operates on sleers of semi-tran
sparent modes of other sessees, essesses of objects so often seese entereds thes
e opposite opposite sessesses is seen studd on our observations seemed the sesse
st of oil-point of openess seems even association of our own thees seems even se
en by observed on objects seen as easies over a landow on a straightly every ses
tood of opacity on the subjective experiments. The observations as sussesses ess
oof to our own operates essential examples of opacity or opaque seeses to see th
e observer of our own terms essees, esesons itself on our own surfaces seesest e
ssential esteemed eeger, colours of our seesest essentials espooldon, as opposed
 to see a seemed see in opposite outlines essesses to some soot, woology, soon o
bserves: so essented to other some of the semi-opaces seems on our observations 
should even executes those of opacity or seen through a sessoot of opacity on ot
her seeses, especially of our object, or even seese the exesono it. To do with o
bjects XIX. Desserse of Optics, of those experiments seems evoke ourselves of ou
r seeseses, essessitions on our observations seems even asserted that other oper
ations seems essential to others of objects seese to see the observations of the
se essential terms equivalent to observes, seen sussesses each other on other ob
jects seen the ese effects of our own operates essential essooce only, to observ
```

I settled on two large patterns: dividing the page into two on a diagonal, and a circle rendered on a contrasting background. The two samples following are in red and green (the latter has a lot of 't'):


### Green and red divided diagonally

```
seems even in some serse especially sees susceptible as equiveded to see the second 
the exession of seeing experiences essentials essees, so sees as seen through each 
other, as essential terms essentially essessity exhibited by esseesions ases especially 
those exeresses is seen seesed to essess in some sessesses, especially sees seems 
to attractest surfaces essential experiences. 333. We essented themselves asserts 
that the terescess assested to essee as subsequents seesest essented to susteen examples, 
 to the thickered see see sees steel, sees seems even some sorted essentises seems 
to the thin theesere sees so strong eyes seems to see the secondary seesest essentials 
to the thicker tendences as see to esses instances especially sees still seeses to 
the thicker than the sesteente edges and sees seesest enseres. Some examples seems 
to attain the thickers of eefliched seese essessities, so sees essential to essessation, 
the tradslation to these eses essentially sees sees sees seems even asserse the essessial 
tott. 222. The two extremes these seeses to see a see in seems essessities, especially 
those of the title' their sceees, esesses to see the second seconds as it essentially 
to the thicker that the thickers of executions seen the ese Seesest essentials esesses 
to the paper. The two extremes of the semi-transparent sessesses in such seesest, 
 the third that the translation of these essentials esseesises in some sessesses, 
 the third to the third to the third see in easies seen through each others. If,were 
the theory of the think to the thicker sees to see the seconds seen through semi-opaque 
mediums to the thicker the translation of seeing technical experiments seems even 
to a tratteretted the two theory of the second elements of eeter instances exespes 
to the thicker than that the third that these seeseses essessities essessities, essentially, 
the two transparent term to the coloured objects seen seeses to see the susceptibility 
of the thin transmitting them to the mottly to see the see to esses instances, seen 
that the two the thin colour to the two theory of seees is seen used to essess in 
the turbit time the terminoty to the term to the tere esteemed been sussessed the 
thickness of the sky takes place the term to the thickered see sees see suesses instead 
 of the thin totally thicker than that the truth of the serse essessions of executions 
that the third that the two strongly thus to be the translated steel, as essees of 
the third to the Thitter the tratth to the Thild transparent sees seems eeterestensine, 
to this the theory the thicker the theory of the theory of the second elements of 
the thin theory the thickness of the thin theory to the thickness of eacheseless 
 to the thicker, the two strongest pressure the term to the thicker exservations 
to the terminology to the third that the white that the two printises seems exested 
to the term (22). The attentive other them to the thicker, the two colours essential 
 to the thicker vertically those of the thinker the theory the thinners see sees 
to the two contrasted to the term (222), to the third that the third than we esess 
to the thicker than that the two extremes the titted the term to the thickered justeseen 
to the thicker that the third then follow the thicker the theory the thickers of 
the theory the temperature. The two that the translator the translator the translators 
```

### A green circle on a red background

```
surface of a stronger sees sees seems even asserted that seen the second elements 
of execution is everynts, seen through semi-opaque mediums seems exhibited ese essential 
 to the surface of semi-transparent external exertions of seeing the second edetest 
elements eses, we see the eyes on the eye essesses in some sort assumed to see a 
serve is also seen that seen steel-wise sees a see the sun shines towards the effects 
of experiments essential to the translation to the terms or even some senses, especially 
sees still perceive the strong great the theory of the terms eres, essessities essessioned 
in the seass of seeing the direction of the thin theory the edges are seen for the 
second editions 3. First Coloured Objects VI. Coloured Stratts on every senses. 333. 
 If the series of the surface of the this theory the phenomena tendered nearest easily 
 exert essential to the thickness of the thinker, the two latter essents effects 
are so seen that the most detical rutural thicker than that the two sessees of exessions 
are emerged from the thin colour attaces the translator observes, the second edges 
mistakes, exhibiting the theory the third think to the attentive observers of nature. 
These exessional examples without the attentive observers them to the eyes as consequence 
of the second class will be the result of this kitt to the light to the same sessesses, 
especially those often according to the thicker that the bright tendency to exemplify 
them is strictly attained that the two strongest state of the phenomena of the second 
series is to be retired, that the tratity to the thicker that the two stresses as 
essential to the thicker, the two foregoing Parts of the two classes themselves essential 
to the eye to the titter, the two latter that the translation of the theory of see 
sees sometimes the third to the other the third that the theory of colours asserts 
susceptible to the strong then the thicker the most beautiful blue is to be seen 
in seen in the third transfacting the thicker than that the thicker vapours of the 
second elementary coloured border than the third than that this attention to these 
examples, to trace the term that the its dark grounds to be the other, especially 
 is perhaps to the theory of the coloured attention to the coloured borders best 
 seems to extend to the same time to trace them, therefore, the translated steel-wise 
so sooner extent it external light, it cannot be attributed to the Treesses as seen 
through a serent of the coloured stratter the appearances that the surface of semi-transparent 
mediums seems to follow the third than the third than that the theory see see sees 
sees subsequently to brilliant coloured figures, and thus the theory of colours seems 
to exemplify these to the attentive observers that the hue of emerges are so seen 
that every second end other the motley. 

 [1] See Note on Pater--"Sisee of the Theory the two essessity especially sees as 
the surface of seeing the eye essessions of exercises of semi-transparent essays 
entirely see in speaking of the surface of semi-transparent mediums seems to see 
the sun shines through each other seese to see a sole of seeing, especially is seen 
in scarcely, essentially essessions, especially sees sees see some serse in some 
```

I also found that my initial idea of rendering different colours by suppressing letters didn't work very well. If I suppressed all letters other than those which look red to me (e, j, s and 3) the resulting output was too constrained to make sense: I tried another strategy, of trying to make red text by suppressing the letters and numbers which are complementary to it, but this didn't produce results which looked red.

Rather than suppressing letters, I tried boosting the weights of the letters for a particular colour: this technique is not, strictly speaking, a lipogram, but it produced the best results in terms of still being readable but having a noticeable synaesthetic effect.



