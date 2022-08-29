---
layout: page
title: Accelerating Neural Audio Synthesis
---

These are audio samples for Václav Volhejn's Master thesis, _Accelerating Neural Audio Synthesis_.
They showcase models that synthesize audio conditioned on pitch and loudness information.
Given an input audio, we first extract its pitch (melody) and loudness.
Then, given only these signals, the model tries to re-create the original sound.
By training the model on a dataset of violin performances, it learns to create violin sounds.

![DDSP architecture diagram](ddsp-architecture.png)

What is exciting about this is that if we have a violin model, we can use it to turn any sound into a violin (listen to the samples in "Violin timbre transfer" below).
Whatever sound we give the model—voice, trumpet, guitar—the model only cares about the pitch and loudness signals.
It then reconstructs the sound as if it were a violin.

The audio samples presented below were used in a subjective listening survey.
The meaning of the rows is:

- _Reference_ is the original audio file.
- _DDSP-full_ is a baseline model from [Engel et al.](https://arxiv.org/abs/2001.04643). About 6M parameters.
- _DDSP-CNN_ is our speed-optimized model (see thesis). About 300k parameters.
- _DDSP-CNN-Tiny_ is a smaller speed-optimized model: it has **less than 2500 parameters**!
- _DDSP-CNN-Q_ is a version of DDSP-CNN quantized using static quantization.
- _DDSP-CNN-Tiny-Q_ is a version of DDSP-CNN-Tiny quantized using static quantization.

## Violin timbre transfer

These models were trained on the same data as the [original DDSP](https://arxiv.org/abs/2001.04643) models, namely four violin pieces by John Gartner that are [accessible](https://musopen.org/music/13574-violin-partita-no-1-bwv-1002/) from the MusOpen royalty free music library.

The reference samples used are from various datasets:
[URMP](https://labsites.rochester.edu/air/projects/URMP.html),
[GuitarSet](https://guitarset.weebly.com/),
[Good-Sounds](https://www.upf.edu/web/mtg/good-sounds),
[SVNote](http://marg.snu.ac.kr/automatic-music-transcription/)
and 
[VocalSet](https://zenodo.org/record/1193957)
.


{::nomarkdown}
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Guitar</th>
      <th>Clarinet</th>
      <th>Flute</th>
      <th>Voice (m)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Reference</th>
      <td><audio src=audio/violin_tt/sample_2_gt.wav controls="" style="width: 175px"></audio></td>
      <td><audio src=audio/violin_tt/sample_8_gt.wav controls="" style="width: 175px"></audio></td>
      <td><audio src=audio/violin_tt/sample_11_gt.wav controls="" style="width: 175px"></audio></td>
      <td><audio src=audio/violin_tt/sample_13_gt.wav controls="" style="width: 175px"></audio></td>
    </tr>
    <tr>
      <th>DDSP-full</th>
      <td><audio src=audio/violin_tt/sample_2_0725-ddspae-2.wav controls="" style="width: 175px"></audio></td>
      <td><audio src=audio/violin_tt/sample_8_0725-ddspae-2.wav controls="" style="width: 175px"></audio></td>
      <td><audio src=audio/violin_tt/sample_11_0725-ddspae-2.wav controls="" style="width: 175px"></audio></td>
      <td><audio src=audio/violin_tt/sample_13_0725-ddspae-2.wav controls="" style="width: 175px"></audio></td>
    </tr>
    <tr>
      <th>DDSP-CNN</th>
      <td><audio src=audio/violin_tt/sample_2_0725-ddspae-cnn-1-rt.wav controls="" style="width: 175px"></audio></td>
      <td><audio src=audio/violin_tt/sample_8_0725-ddspae-cnn-1-rt.wav controls="" style="width: 175px"></audio></td>
      <td><audio src=audio/violin_tt/sample_11_0725-ddspae-cnn-1-rt.wav controls="" style="width: 175px"></audio></td>
      <td><audio src=audio/violin_tt/sample_13_0725-ddspae-cnn-1-rt.wav controls="" style="width: 175px"></audio></td>
    </tr>
    <tr>
      <th>DDSP-CNN-Tiny</th>
      <td><audio src=audio/violin_tt/sample_2_0809-ddspae-cnn-5-rt.wav controls="" style="width: 175px"></audio></td>
      <td><audio src=audio/violin_tt/sample_8_0809-ddspae-cnn-5-rt.wav controls="" style="width: 175px"></audio></td>
      <td><audio src=audio/violin_tt/sample_11_0809-ddspae-cnn-5-rt.wav controls="" style="width: 175px"></audio></td>
      <td><audio src=audio/violin_tt/sample_13_0809-ddspae-cnn-5-rt.wav controls="" style="width: 175px"></audio></td>
    </tr>
    <tr>
      <th>DDSP-CNN-Q</th>
      <td><audio src=audio/violin_tt/sample_2_0725-ddspae-cnn-1-rtq.wav controls="" style="width: 175px"></audio></td>
      <td><audio src=audio/violin_tt/sample_8_0725-ddspae-cnn-1-rtq.wav controls="" style="width: 175px"></audio></td>
      <td><audio src=audio/violin_tt/sample_11_0725-ddspae-cnn-1-rtq.wav controls="" style="width: 175px"></audio></td>
      <td><audio src=audio/violin_tt/sample_13_0725-ddspae-cnn-1-rtq.wav controls="" style="width: 175px"></audio></td>
    </tr>
    <tr>
      <th>DDSP-CNN-Tiny-Q</th>
      <td><audio src=audio/violin_tt/sample_2_0809-ddspae-cnn-5-rtq.wav controls="" style="width: 175px"></audio></td>
      <td><audio src=audio/violin_tt/sample_8_0809-ddspae-cnn-5-rtq.wav controls="" style="width: 175px"></audio></td>
      <td><audio src=audio/violin_tt/sample_11_0809-ddspae-cnn-5-rtq.wav controls="" style="width: 175px"></audio></td>
      <td><audio src=audio/violin_tt/sample_13_0809-ddspae-cnn-5-rtq.wav controls="" style="width: 175px"></audio></td>
    </tr>
  </tbody>
</table>
{:/nomarkdown}

{::nomarkdown}
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Saxophone</th>
      <th>Guitar</th>
      <th>Voice (f)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Reference</th>
      <td><audio src=audio/violin_tt/sample_20_gt.wav controls="" style="width: 175px"></audio></td>
      <td><audio src=audio/violin_tt/sample_23_gt.wav controls="" style="width: 175px"></audio></td>
      <td><audio src=audio/violin_tt/sample_29_gt.wav controls="" style="width: 175px"></audio></td>
    </tr>
    <tr>
      <th>DDSP-full</th>
      <td><audio src=audio/violin_tt/sample_20_0725-ddspae-2.wav controls="" style="width: 175px"></audio></td>
      <td><audio src=audio/violin_tt/sample_23_0725-ddspae-2.wav controls="" style="width: 175px"></audio></td>
      <td><audio src=audio/violin_tt/sample_29_0725-ddspae-2.wav controls="" style="width: 175px"></audio></td>
    </tr>
    <tr>
      <th>DDSP-CNN</th>
      <td><audio src=audio/violin_tt/sample_20_0725-ddspae-cnn-1-rt.wav controls="" style="width: 175px"></audio></td>
      <td><audio src=audio/violin_tt/sample_23_0725-ddspae-cnn-1-rt.wav controls="" style="width: 175px"></audio></td>
      <td><audio src=audio/violin_tt/sample_29_0725-ddspae-cnn-1-rt.wav controls="" style="width: 175px"></audio></td>
    </tr>
    <tr>
      <th>DDSP-CNN-Tiny</th>
      <td><audio src=audio/violin_tt/sample_20_0809-ddspae-cnn-5-rt.wav controls="" style="width: 175px"></audio></td>
      <td><audio src=audio/violin_tt/sample_23_0809-ddspae-cnn-5-rt.wav controls="" style="width: 175px"></audio></td>
      <td><audio src=audio/violin_tt/sample_29_0809-ddspae-cnn-5-rt.wav controls="" style="width: 175px"></audio></td>
    </tr>
    <tr>
      <th>DDSP-CNN-Q</th>
      <td><audio src=audio/violin_tt/sample_20_0725-ddspae-cnn-1-rtq.wav controls="" style="width: 175px"></audio></td>
      <td><audio src=audio/violin_tt/sample_23_0725-ddspae-cnn-1-rtq.wav controls="" style="width: 175px"></audio></td>
      <td><audio src=audio/violin_tt/sample_29_0725-ddspae-cnn-1-rtq.wav controls="" style="width: 175px"></audio></td>
    </tr>
    <tr>
      <th>DDSP-CNN-Tiny-Q</th>
      <td><audio src=audio/violin_tt/sample_20_0809-ddspae-cnn-5-rtq.wav controls="" style="width: 175px"></audio></td>
      <td><audio src=audio/violin_tt/sample_23_0809-ddspae-cnn-5-rtq.wav controls="" style="width: 175px"></audio></td>
      <td><audio src=audio/violin_tt/sample_29_0809-ddspae-cnn-5-rtq.wav controls="" style="width: 175px"></audio></td>
    </tr>
  </tbody>
</table>
{:/nomarkdown}


## Violin

These are the violin models from above, but this time they are used to reconstruct the original data.


{::nomarkdown}
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Reference</th>
      <td><audio src=audio/violin/sample_6_gt.wav controls="" style="width: 175px"></audio></td>
      <td><audio src=audio/violin/sample_8_gt.wav controls="" style="width: 175px"></audio></td>
      <td><audio src=audio/violin/sample_10_gt.wav controls="" style="width: 175px"></audio></td>
    </tr>
    <tr>
      <th>DDSP-full</th>
      <td><audio src=audio/violin/sample_6_0725-ddspae-2.wav controls="" style="width: 175px"></audio></td>
      <td><audio src=audio/violin/sample_8_0725-ddspae-2.wav controls="" style="width: 175px"></audio></td>
      <td><audio src=audio/violin/sample_10_0725-ddspae-2.wav controls="" style="width: 175px"></audio></td>
    </tr>
    <tr>
      <th>DDSP-CNN</th>
      <td><audio src=audio/violin/sample_6_0725-ddspae-cnn-1-rt.wav controls="" style="width: 175px"></audio></td>
      <td><audio src=audio/violin/sample_8_0725-ddspae-cnn-1-rt.wav controls="" style="width: 175px"></audio></td>
      <td><audio src=audio/violin/sample_10_0725-ddspae-cnn-1-rt.wav controls="" style="width: 175px"></audio></td>
    </tr>
    <tr>
      <th>DDSP-CNN-Tiny</th>
      <td><audio src=audio/violin/sample_6_0809-ddspae-cnn-5-rt.wav controls="" style="width: 175px"></audio></td>
      <td><audio src=audio/violin/sample_8_0809-ddspae-cnn-5-rt.wav controls="" style="width: 175px"></audio></td>
      <td><audio src=audio/violin/sample_10_0809-ddspae-cnn-5-rt.wav controls="" style="width: 175px"></audio></td>
    </tr>
    <tr>
      <th>DDSP-CNN-Q</th>
      <td><audio src=audio/violin/sample_6_0725-ddspae-cnn-1-rtq.wav controls="" style="width: 175px"></audio></td>
      <td><audio src=audio/violin/sample_8_0725-ddspae-cnn-1-rtq.wav controls="" style="width: 175px"></audio></td>
      <td><audio src=audio/violin/sample_10_0725-ddspae-cnn-1-rtq.wav controls="" style="width: 175px"></audio></td>
    </tr>
    <tr>
      <th>DDSP-CNN-Tiny-Q</th>
      <td><audio src=audio/violin/sample_6_0809-ddspae-cnn-5-rtq.wav controls="" style="width: 175px"></audio></td>
      <td><audio src=audio/violin/sample_8_0809-ddspae-cnn-5-rtq.wav controls="" style="width: 175px"></audio></td>
      <td><audio src=audio/violin/sample_10_0809-ddspae-cnn-5-rtq.wav controls="" style="width: 175px"></audio></td>
    </tr>
  </tbody>
</table>
{:/nomarkdown}


## Trumpet

These models were trained on trumpet performances from the [URMP dataset](https://labsites.rochester.edu/air/projects/URMP.html).


{::nomarkdown}
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Reference</th>
      <td><audio src=audio/trumpet/sample_0_gt.wav controls="" style="width: 175px"></audio></td>
      <td><audio src=audio/trumpet/sample_2_gt.wav controls="" style="width: 175px"></audio></td>
      <td><audio src=audio/trumpet/sample_4_gt.wav controls="" style="width: 175px"></audio></td>
    </tr>
    <tr>
      <th>DDSP-full</th>
      <td><audio src=audio/trumpet/sample_0_0805-ddspae.wav controls="" style="width: 175px"></audio></td>
      <td><audio src=audio/trumpet/sample_2_0805-ddspae.wav controls="" style="width: 175px"></audio></td>
      <td><audio src=audio/trumpet/sample_4_0805-ddspae.wav controls="" style="width: 175px"></audio></td>
    </tr>
    <tr>
      <th>DDSP-CNN</th>
      <td><audio src=audio/trumpet/sample_0_0804-ddspae-cnn-3-rt.wav controls="" style="width: 175px"></audio></td>
      <td><audio src=audio/trumpet/sample_2_0804-ddspae-cnn-3-rt.wav controls="" style="width: 175px"></audio></td>
      <td><audio src=audio/trumpet/sample_4_0804-ddspae-cnn-3-rt.wav controls="" style="width: 175px"></audio></td>
    </tr>
    <tr>
      <th>DDSP-CNN-Tiny</th>
      <td><audio src=audio/trumpet/sample_0_0809-ddspae-cnn-4-rt.wav controls="" style="width: 175px"></audio></td>
      <td><audio src=audio/trumpet/sample_2_0809-ddspae-cnn-4-rt.wav controls="" style="width: 175px"></audio></td>
      <td><audio src=audio/trumpet/sample_4_0809-ddspae-cnn-4-rt.wav controls="" style="width: 175px"></audio></td>
    </tr>
    <tr>
      <th>DDSP-CNN-Q</th>
      <td><audio src=audio/trumpet/sample_0_0804-ddspae-cnn-3-rtq.wav controls="" style="width: 175px"></audio></td>
      <td><audio src=audio/trumpet/sample_2_0804-ddspae-cnn-3-rtq.wav controls="" style="width: 175px"></audio></td>
      <td><audio src=audio/trumpet/sample_4_0804-ddspae-cnn-3-rtq.wav controls="" style="width: 175px"></audio></td>
    </tr>
    <tr>
      <th>DDSP-CNN-Tiny-Q</th>
      <td><audio src=audio/trumpet/sample_0_0809-ddspae-cnn-4-rtq.wav controls="" style="width: 175px"></audio></td>
      <td><audio src=audio/trumpet/sample_2_0809-ddspae-cnn-4-rtq.wav controls="" style="width: 175px"></audio></td>
      <td><audio src=audio/trumpet/sample_4_0809-ddspae-cnn-4-rtq.wav controls="" style="width: 175px"></audio></td>
    </tr>
  </tbody>
</table>
{:/nomarkdown}


