import torch
from torch import nn

class Finder(nn.Module):
  def __init__(self, fv_num, qe_ptfile):
    super(Finder, self).__init__()
    
    self.net = nn.Sequential(
    nn.Linear(fv_num+3, 256, bias=True, dtype=torch.float64),  # set up first FC layer,
    nn.LayerNorm(256, dtype=torch.float64),
    # nn.BatchNorm1d(256,  dtype=torch.float64),
    nn.ReLU(),
    nn.Linear(256, 3, bias=True, dtype=torch.float64),
    )
    
    for m in self.modules():
      if isinstance(m, nn.Linear):
          torch.nn.init.xavier_uniform_(m.weight.data)
    
    self.qe = QualityEstimator(fv_num)
    self.qe.load_state_dict(torch.load(qe_ptfile))
    self.qe.eval()
    
  def forward(self, fv, ref_settings):
    # now we can reshape `c` and `f` to 2D and concat them
    # combined = torch.cat((fv.view(fv.size(0), -1),
    #                       iso.view(iso.size(0), -1),
    #                       ss.view(ss.size(0), -1),
    #                       aperts.view(aperts.size(0), -1)), dim=-1)
    iso_r, ss_r, aperts_r = ref_settings
    combined = torch.cat((fv.view(fv.size(0), -1),
                          iso_r.view(iso_r.size(0), -1),
                          ss_r.view(ss_r.size(0), -1),
                          aperts_r.view(aperts_r.size(0), -1)), dim=-1)
    finded = self.net(combined)

    predicted_scores, changed_fv = self.qe(fv, ref_settings, (finded[:,0], finded[:,0], finded[:,0]))

    return finded, predicted_scores
    
    # fv, present setting
  
class QualityEstimator(nn.Module):
  def __init__(self, fv_num):
    super(QualityEstimator, self).__init__()
    # self.gen_fv = nn.Sequential(
    #   nn.Linear(fv_num+3, fv_num//2, bias=True, dtype=torch.float64),
    #   nn.LayerNorm(fv_num//2, dtype=torch.float64),
    #   nn.ReLU(),
    #   nn.Linear(fv_num//2, fv_num//2//2, bias=True, dtype=torch.float64),
    #   nn.LayerNorm(fv_num//2//2, dtype=torch.float64),
    #   nn.ReLU(),
    #   nn.Linear(fv_num//2//2, fv_num//2, bias=True, dtype=torch.float64),
    #   nn.LayerNorm(fv_num//2, dtype=torch.float64),
    #   nn.ReLU(),
    #   nn.Linear(fv_num//2, fv_num, bias=True, dtype=torch.float64),
    #   nn.LayerNorm(fv_num, dtype=torch.float64),
    #   nn.ReLU()
    # )
    self.gen_fv = nn.Sequential(
      nn.Linear(fv_num+6, fv_num, bias=True, dtype=torch.float64),
      nn.LayerNorm(fv_num, dtype=torch.float64),
      nn.ReLU(),
    )
    self.net = nn.Sequential(
    nn.Linear(fv_num, 512, bias=True, dtype=torch.float64),  # set up first FC layer,
    nn.LayerNorm(512, dtype=torch.float64),
    # nn.BatchNorm1d(256,  dtype=torch.float64),
    nn.ReLU(),
    nn.Linear(512, 256, bias=True, dtype=torch.float64),
    nn.LayerNorm(256, dtype=torch.float64),
    # nn.BatchNorm1d(32,  dtype=torch.float64),
    nn.ReLU(),
    nn.Linear(256, 128, bias=True, dtype=torch.float64),
    nn.LayerNorm(128, dtype=torch.float64),
    nn.ReLU(),
    nn.Linear(128, 32, bias=True, dtype=torch.float64),
    nn.LayerNorm(32, dtype=torch.float64),
    nn.ReLU(),
    nn.Linear(32, 1, bias=True, dtype=torch.float64)
    )
    
    for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)

  def forward(self, fv, ref_settings, change_settings):
    # change_settings = (iso, ss, aperts)
    # ref_settings = (iso, ss, aperts)
    # now we can reshape `c` and `f` to 2D and concat them
    iso_r, ss_r, aperts_r = ref_settings
    iso,ss, aperts = change_settings
    combined = torch.cat((fv.view(fv.size(0), -1),
                          iso_r.view(iso_r.size(0), -1),
                          ss_r.view(ss_r.size(0), -1),
                          aperts_r.view(aperts_r.size(0), -1),
                          iso.view(iso.size(0), -1),
                          ss.view(ss.size(0), -1),
                          aperts.view(aperts.size(0), -1)), dim=-1)
    gened_fv = self.gen_fv(combined)
    out = self.net(gened_fv)
    return out, gened_fv
  
class FV_Generator(nn.Module): # CAE
  def __init__(self, fv_num):
    super(FV_Generator, self).__init__()
    
    self.encoder = nn.Sequential(
      nn.Linear(fv_num+6, fv_num//2, bias=True, dtype=torch.float64),
      nn.LayerNorm(fv_num//2, dtype=torch.float64),
      nn.ReLU(),
      nn.Linear(fv_num//2, fv_num//2//2, bias=True, dtype=torch.float64),
      nn.LayerNorm(fv_num//2//2, dtype=torch.float64),
      nn.ReLU()
    )
    
    self.decoder = nn.Sequential(
      nn.Linear(fv_num//2//2+6, fv_num//2, bias=True, dtype=torch.float64),
      nn.LayerNorm(fv_num//2, dtype=torch.float64),
      nn.ReLU(),
      nn.Linear(fv_num//2, fv_num, bias=True, dtype=torch.float64),
      nn.LayerNorm(fv_num, dtype=torch.float64),
      nn.ReLU(),
    )
    
  def forward(self, fv, ref_settings, change_settings):
    iso_r, ss_r, aperts_r = ref_settings
    iso, ss, aperts = change_settings
    combined = torch.cat((fv.view(fv.size(0), -1),
                            iso_r.view(iso_r.size(0), -1),
                            ss_r.view(ss_r.size(0), -1),
                            aperts_r.view(aperts_r.size(0), -1),
                            iso.view(iso.size(0), -1),
                            ss.view(ss.size(0), -1),
                            aperts.view(aperts.size(0), -1)), dim=-1)
    encoded = self.encoder(combined)
    combined_enc = torch.cat((encoded,
                            iso_r.view(iso_r.size(0), -1),
                            ss_r.view(ss_r.size(0), -1),
                            aperts_r.view(aperts_r.size(0), -1),
                            iso.view(iso.size(0), -1),
                            ss.view(ss.size(0), -1),
                            aperts.view(aperts.size(0), -1)), dim=-1)
    gened_fv = self.decoder(combined_enc)
    return gened_fv

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True, dtype=torch.float64)
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm(hidden_dim, dtype=torch.float64)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim, bias=True, dtype=torch.float64)
        self.fc_logv = nn.Linear(hidden_dim, latent_dim, bias=True, dtype=torch.float64)
    def forward(self, x):
        
        h = self.relu(self.fc1(x))
        h = self.ln(h)
        z_mean = self.fc_mean(h)
        z_logv = self.fc_logv(h)
        return z_mean, z_logv
    
class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True, dtype=torch.float64)
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm(hidden_dim, dtype=torch.float64)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=True, dtype=torch.float64)
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self, z):
        h = self.relu(self.fc1(z))
        h = self.ln(h)
        x_recon = self.sigmoid(self.fc2(h))
        return x_recon
      
class CVAE(nn.Module):
    def __init__(self, fv_num):
        super(CVAE, self).__init__()
        
        # 인코더, 디코더 정의
        self.encoder = Encoder(fv_num+3, fv_num//2, fv_num//4) 
        self.decoder = Decoder(fv_num//4+3 , fv_num//2, fv_num)

    def forward(self, fv, cur_settings, next_settings):  #TODO: concat 해서 넣게 바꿔줘야함.
         # 인코더 forward
        iso_r, ss_r, aperts_r = cur_settings
        iso, ss, aperts = next_settings
        combined = torch.cat((fv.view(fv.size(0), -1),
                              iso_r.view(iso_r.size(0), -1),
                              ss_r.view(ss_r.size(0), -1),
                              aperts_r.view(aperts_r.size(0), -1)),
                              dim=-1)
        z_mean, z_logv = self.encoder(combined)
        z = self.reparameterize(z_mean, z_logv)
        combined_enc = torch.cat((z,                    
                              iso.view(iso.size(0), -1),
                              ss.view(ss.size(0), -1),
                              aperts.view(aperts.size(0), -1)), dim=-1)
        gened_fv = self.decoder(combined_enc)
        # z_mean, z_logv = self.encoder(fv, cur_settings)
        # z = self.reparameterize(z_mean, z_logv)
        
        # # 디코더 forward
        # x_recon = self.decoder(z, next_settings)
        
        return gened_fv, z_mean, z_logv
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)  
        eps = torch.randn_like(std)
        return mu + eps*std

    def inference(self, z, conditions):
        return self.decoder(z, conditions)