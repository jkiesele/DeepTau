#!/bin/env python

from DeepJetCore.evaluation import makePlots_async,makeROCs_async
from argparse import ArgumentParser




parser = ArgumentParser('Apply a model to a (test) sample and create friend trees to inject it inthe original ntuple')
parser.add_argument('inputDir')
args = parser.parse_args()

infile=args.inputDir+'/tree_association.txt'
outdir=args.inputDir+'/'



makeROCs_async(intextfile=infile, 
                       name_list=          ['cut based', 'DNN'], 
                       probabilities_list= ['prob_isTau', '1 - demetraIsolation/(demetraIsolation+1)'],
                       truths_list=        'isTau', 
                       vetos_list=         'isNoTau', 
                       colors_list='auto,dashed', 
                       outpdffile=outdir+'ROC_ID_only_dscompare.pdf', 
                       cuts=['isRecTau && recTau_pt>20'],
                       treename='tree',
                       invalidlist='isRecTau<0.5',
                       firstcomment='rec #tau jet p_{T}>20 GeV')



makeROCs_async(intextfile=infile, 
                       name_list=          ['ROC'], 
                       probabilities_list= 'prob_isTau', 
                       truths_list=        'isTau', 
                       vetos_list=         'isNoTau', 
                       colors_list='auto,dashed', 
                       outpdffile=outdir+'ROC_ID_only_all.pdf', 
                       cuts=['isRecTau>0 && recJet_pt>20'],
                       treename='tree',
                       firstcomment='rec #tau jet p_{T}>20 GeV')

makeROCs_async(intextfile=infile, 
                       name_list=          ['rec #tau','any'], 
                       probabilities_list= 'prob_isTau', 
                       truths_list=        'isTau', 
                       vetos_list=         'isNoTau', 
                       colors_list='auto,dashed', 
                       outpdffile=outdir+'ROC_ID_rec_incl_all.pdf', 
                       cuts=['recJet_pt>20'],
                       invalidlist=['isRecTau<0.5',''],
                       treename='tree',
                       firstcomment='jet p_{T}>20 GeV')

makeROCs_async(intextfile=infile, 
                       name_list=          ['p_{T}^{#tau}=[20,30] GeV',
                                            'p_{T}^{#tau}=[30,40] GeV',
                                            'p_{T}^{#tau}=[40,60] GeV',
                                            'p_{T}^{#tau}=[60,100] GeV',
                                            'p_{T}^{#tau}>100 GeV'], 
                       probabilities_list= 'prob_isTau', 
                       truths_list=        'isTau', 
                       vetos_list=         'isNoTau', 
                       colors_list='auto,dashed', 
                       outpdffile=outdir+'ROC_ID_only_pt.pdf', 
                       cuts=['isRecTau && recTau_pt>20 && recTau_pt<30',
                             'isRecTau && recTau_pt>30 && recTau_pt<40',
                             'isRecTau && recTau_pt>40 && recTau_pt<60',
                             'isRecTau && recTau_pt>60 && recTau_pt<100',
                             'isRecTau && recTau_pt>100',
                             ],
                       treename='tree')

makeROCs_async(intextfile=infile, 
                       name_list=          ['p_{T}^{jet}=[20,30] GeV',
                                            'p_{T}^{jet}=[30,40] GeV',
                                            'p_{T}^{jet}=[40,60] GeV',
                                            'p_{T}^{jet}=[60,100] GeV',
                                            'p_{T}^{jet}>100 GeV'], 
                       probabilities_list= 'prob_isTau', 
                       truths_list=        'isTau', 
                       vetos_list=         'isNoTau', 
                       colors_list='auto,dashed', 
                       outpdffile=outdir+'ROC_ID_rec_incl_pt.pdf', 
                       cuts=['recJet_pt>20 && recJet_pt<30',
                             'recJet_pt>30 && recJet_pt<40',
                             'recJet_pt>40 && recJet_pt<60',
                             'recJet_pt>60 && recJet_pt<100',
                             'recJet_pt>100',
                             ],
                       invalidlist=5*['isRecTau<0.5'],
                       treename='tree')