import sys
# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py
sys.path.append('/home/ubuntu/Kaggle_Pytorch_TGS/semantic-segmentation-pytorch')
from models import *
from torchvision.transforms import Normalize

def build_segmentation_model(in_arch="resnet101_dilated8",out_arch="upernet",droppout=0.1):
    
    '''
    So we allow 3 versions of the model for now: A Resnet 50 101 with Upernet Decoder pre trained on Imagenet 
    also a small model 
    
    '''
    #First we build two verison of the model for MEan Teacher or not. 
    builder = ModelBuilder()
    #Define Encoder
    net_encoder = builder.build_encoder(
        arch=in_arch,
        #fc_dim=2048
    )

    #Define Decoder
    net_decoder = builder.build_decoder(
        arch=out_arch,
        fc_dim=2048,
        #weights here lets us load our own weights neat
        num_class=2)

    net_encoder_ema = builder.build_encoder(
        arch=in_arch,
        #fc_dim=2048
    )

    #Define Decoder
    net_decoder_ema = builder.build_decoder(
        arch=out_arch,
        fc_dim=2048,
        #weights here lets us load our own weights neat
        num_class=2)
    
    class SegmentationModule(SegmentationModuleBase):
        def __init__(self, net_enc, net_dec,drop=0,size=101):
            super(SegmentationModule, self).__init__()
            self.encoder = net_enc
            self.decoder = net_dec
            self.drop=drop
            self.size=size
            
        def forward(self, feed_dict, *, segSize=None):
            inpu=feed_dict['img_data']
            
            encode= self.encoder(inpu, return_feature_maps=True)

            if self.drop>0:
                encode[0]=nn.Dropout(self.drop)(encode[0])
                encode[1]=nn.Dropout(self.drop)(encode[1])
                encode[2]=nn.Dropout(self.drop)(encode[2])
                encode[3]=nn.Dropout(self.drop)(encode[3])

            pred = self.decoder(encode)
            pred = nn.functional.upsample(pred, size=self.size, mode='bilinear', align_corners=True)
            #Lovasz Softmax needs Sonftmax inputs.
            #pred = nn.functional.softmax(pred, dim=1)

            return pred
        
    segmentation_ema=SegmentationModule(
            net_encoder_ema, net_decoder_ema,drop=droppout)


        
    segmentation_ema=segmentation_ema.cuda()

    #Set up the complete model
    segmentation_module = SegmentationModule(
                net_encoder, net_decoder,drop=droppout)
    segmentation_module=segmentation_module.cuda()
    
    for param,param2 in zip(segmentation_ema.parameters(),segmentation_module.parameters()):
        param.data=param2.data
        
    for param in segmentation_ema.parameters():
        param.detach_()
        
    return segmentation_module,segmentation_ema
    
        
    