
from DeepJetCore.TrainData import TrainData
from DeepJetCore.TrainData import fileTimeOut
import numpy 

class TrainData_deeptau(TrainData):
    def __init__(self):
        TrainData.__init__(self)

        self.treename="tree" #input root tree name
        
        self.truthclasses=['isTau','isNoTau'] #truth classes for classification
        
        self.weightbranchX='recTau_pt' #needs to be specified
        self.weightbranchY='recTau_eta' #needs to be specified
        
        self.referenceclass='isTau'
        #this removes everything that is not a recTau, which has pt>0, and eta!=0
        self.weight_binX = numpy.array([0,5,20,22.5,25,27.5,30,35,40,45,50,60,70,80,100,150,200,250,500,1e5],dtype=float) 
        self.weight_binY = numpy.array([-4,-3,-2,-1, -1e-4,0,1e-4, 1,2,3,4],dtype=float) 
        
        
        #globals
        self.addBranches([
            #'isRecTau',
            'numPileUp', 
            'recTauDecayMode', 
            'recTau_pt', 
            'recTau_eta', 
            'recTau_phi', 
            'recTau_M', 
            'recTauVtxZ', 
            'recImpactParamPCA_x', 
            'recImpactParamPCA_y', 
            'recImpactParamPCA_z', 
            'recImpactParam', 
            'recImpactParamSign', 
            'recImpactParam3D', 
            'recImpactParamSign3D', 
            'hasRecDecayVertex', 
            'recDecayDist_x', 
            'recDecayDist_y', 
            'recDecayDist_z', 
            'recDecayDistSign', 
            'recTauPtWeightedDetaStrip', 
            'recTauPtWeightedDphiStrip', 
            'recTauPtWeightedDrSignal', 
            'recTauPtWeightedDrIsolation', 
            'recTauNphoton', 
            'recTauEratio', 
            'recTauLeadingTrackChi2', 
            'recTauNphotonSignal', 
            'recTauNphotonIso', 
            'recJet_pt', 
            'recJet_eta', 
            'recJet_mass', 
            'recJetLooseId', 
            'nCpfcan', 
            'nNpfcand',  
            ])
       
        self.addBranches([
            'Cpfcan_pt',
            'Cpfcan_eta',
            'Cpfcan_ptrel',
            'Cpfcan_erel',
            'Cpfcan_deltaR',
            'Cpfcan_puppiw',
            'Cpfcan_VTX_ass',
            'Cpfcan_fromPV',
            'Cpfcan_vertex_rho',
            'Cpfcan_vertex_phirel',
            'Cpfcan_vertex_etarel',
            'Cpfcan_dz',
            'Cpfcan_dxy',
            'Cpfcan_dxyerrinv',
            'Cpfcan_dxysig',
            'Cpfcan_BtagPf_trackMomentum',
            'Cpfcan_BtagPf_trackEtaRel',
            'Cpfcan_BtagPf_trackPtRel',
            'Cpfcan_BtagPf_trackPPar',
            'Cpfcan_BtagPf_trackDeltaR',
            'Cpfcan_BtagPf_trackPtRatio',
            'Cpfcan_BtagPf_trackPParRatio',
            'Cpfcan_BtagPf_trackSip3dVal',
            'Cpfcan_BtagPf_trackSip3dSig',
            'Cpfcan_BtagPf_trackSip2dVal',
            'Cpfcan_BtagPf_trackSip2dSig',
            'Cpfcan_BtagPf_trackJetDistVal',
            'Cpfcan_isMu',
            'Cpfcan_isEl',
            'Cpfcan_pdgID',
            'Cpfcan_charge',
            'Cpfcan_lostInnerHits',
            'Cpfcan_numberOfPixelHits',
            'Cpfcan_chi2',
            'Cpfcan_quality',
        ], 20)
        
        self.addBranches([
            'Npfcan_pt',
            'Npfcan_eta',
            'Npfcan_ptrel',
            'Npfcan_erel',
            'Npfcan_puppiw',
            'Npfcan_phirel',
            'Npfcan_etarel',
            'Npfcan_deltaR',
            'Npfcan_isGamma',
            'Npfcan_HadFrac',
        ], 40)
        
        self.addBranches(['recTau_pt','recJet_pt'
        ])
        
        self.regtruth='genTau_pt'

        self.regressiontargetclasses=['pt']
        
        
        self.registerBranches(['genTau_pt']) #list of branches to be used 
        
        self.registerBranches(self.truthclasses)
        
        
        #call this at the end
        self.reduceTruth(None)
        
    
    def readFromRootFile(self,filename,TupleMeanStd, weighter):
    
        # this function defines how to convert the root ntuple to the training format
        # options are not yet described here
        from DeepJetCore.preprocessing import MeanNormApply, MeanNormZeroPad, MeanNormZeroPadParticles
        
        import ROOT
        
        fileTimeOut(filename,120) #give eos a minute to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get("tree")
        self.nsamples=tree.GetEntries()
        
        npy_array = self.readTreeFromRootToTuple(filename)
        
        truthtuple = npy_array[self.truthclasses]
        
        alltruth=self.reduceTruth(truthtuple)
        alltruept=npy_array[self.regtruth]
        
        # user code
        x_global = MeanNormZeroPad(
            filename,None,
            [self.branches[0]],
            [self.branchcutoffs[0]],self.nsamples
        )
        
        x_cpf = MeanNormZeroPadParticles(
            filename,None,
            self.branches[1],
            self.branchcutoffs[1],self.nsamples
        )
        
        x_npf = MeanNormZeroPadParticles(
            filename,None,
            self.branches[2],
            self.branchcutoffs[2],self.nsamples
        )
        
        x_recopts = MeanNormZeroPad(
            filename,None,
            [self.branches[3]],
            [self.branchcutoffs[3]],self.nsamples
        )
        
        
        nold=self.nsamples
        
        self.x=[x_global,x_cpf,x_npf,x_recopts] # list of feature numpy arrays
        self.y=[alltruth,alltruept] # list of target numpy arrays (truth)
        self.w=[] # list of weight arrays. One for each truth target
        self._normalize_input_(weighter,npy_array)
        
        print('reduced to ',self.nsamples , 'of', nold)
        


class TrainData_deeptau_rec(TrainData_deeptau):
    def __init__(self):
        TrainData_deeptau.__init__(self)
        
        self.weightbranchX='recJet_pt' #needs to be specified
        self.weightbranchY='recJet_eta' #needs to be specified
        
        self.referenceclass='isTau'
        #this removes everything that is not a recTau, which has pt>0, and eta!=0
        self.weight_binX = numpy.array([0,20,22.5,25,27.5,30,35,40,45,50,60,70,80,100,150,200,250,500,1e5],dtype=float) 
        self.weight_binY = numpy.array([-4,-3,-2,-1,0, 1,2,3,4],dtype=float) 
        
        
        
        
        
        
        
        
        
        
        
        