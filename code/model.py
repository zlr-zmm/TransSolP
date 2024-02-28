import torch.nn as nn
import torch
class CNN_Transformer_BiLSTM(nn.Module):
    def __init__(self, input_dim1, input_dim2,  num_classes):
        super(CNN_Transformer_BiLSTM, self).__init__()
        self.multihead_attention1 = nn.MultiheadAttention(input_dim1, num_heads=4)
        self.layer_norm1 = nn.LayerNorm(input_dim1)
        self.feed_forward1 = nn.Sequential(
            nn.Linear(input_dim1, 4 * input_dim1),
            nn.ReLU(),
            nn.Linear(4 * input_dim1, input_dim1)
        )
        self.layer_norm12 = nn.LayerNorm(input_dim1)
        self.dropout1 = nn.Dropout(0.1)
        self.fc1 = nn.Linear(input_dim1, input_dim1)
        self.fc12 = nn.Linear(input_dim1, num_classes)

        self.bilstm1 = nn.LSTM(input_dim1, 512, batch_first=True, bidirectional=True)

        self.residual_conv1 = nn.Conv1d(input_dim1, input_dim1, kernel_size=1)
        self.conv11 = nn.Conv1d(input_dim1, input_dim1, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv12 = nn.Conv1d(input_dim1, input_dim1, kernel_size=3, padding=1)

        # input_dim2
        self.multihead_attention2 = nn.MultiheadAttention(input_dim2, num_heads=4)
        self.layer_norm2 = nn.LayerNorm(input_dim2)
        self.feed_forward2 = nn.Sequential(
            nn.Linear(input_dim2, 4 * input_dim2),
            nn.ReLU(),
            nn.Linear(4 * input_dim2, input_dim2)
        )
        self.layer_norm22 = nn.LayerNorm(input_dim2)
        self.dropout2 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(input_dim2, input_dim2)
        self.fc22 = nn.Linear(input_dim2, num_classes)

        self.bilstm2 = nn.LSTM(input_dim2, 512, batch_first=True, bidirectional=True)

        self.residual_conv2 = nn.Conv1d(input_dim2, input_dim2, kernel_size=1)
        self.conv21 = nn.Conv1d(input_dim2, input_dim2, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.conv22 = nn.Conv1d(input_dim2, input_dim2, kernel_size=3, padding=1)

        self.fc_final1 = nn.Linear(10240, 1024)
        self.fc_final2 = nn.Linear(1024, num_classes)

    def prott5(self, x,mask=None):

        x11 = x.permute(0, 2, 1)
        residual1 = self.residual_conv1(x11)
        out_cnn1 = self.conv11(x11)
        out_cnn1 = self.relu1(out_cnn1)
        out_cnn1 = self.conv12(out_cnn1)
        out_cnn1 += residual1
        out_cnn0 = out_cnn1.squeeze(2)
        out_cnn1 = self.relu1(out_cnn1)
        out_cnn1 = torch.mean(out_cnn1, dim=2)

        attened, _ = self.multihead_attention1(x, x, x, attn_mask=mask)
        residual1 = x + self.dropout1(attened)
        norm1 = self.layer_norm1(residual1)
        feed_forward_out = self.feed_forward1(norm1)
        residual2 = norm1 + self.dropout1(feed_forward_out)
        norm2 = self.layer_norm12(residual2)
        norm1 = norm2.squeeze(1)
        out_transformer = self.fc1(norm2)
        out_transformer = self.dropout1(out_transformer)
        out_transformer = out_transformer.squeeze(dim=1)

        x_biLstm1 = x.to(torch.float32)
        _, (hidden_bilstm1, _) = self.bilstm1(x_biLstm1)
        final1 = torch.cat((out_cnn1, out_transformer, hidden_bilstm1[-2], hidden_bilstm1[-1]), dim=1)
        return out_cnn0, norm1, torch.cat((hidden_bilstm1[-2], hidden_bilstm1[-1]), dim=1),final1

    def esm(self, x, mask=None):
        x11 = x.permute(0, 2, 1)
        residual1 = self.residual_conv2(x11)
        out_cnn1 = self.conv21(x11)
        out_cnn1 = self.relu2(out_cnn1)
        out_cnn1 = self.conv22(out_cnn1)
        out_cnn1 += residual1
        out_cnn0 = out_cnn1.squeeze(2)
        out_cnn1 = self.relu2(out_cnn1)
        out_cnn1 = torch.mean(out_cnn1, dim=2)

        attened, _ = self.multihead_attention2(x, x, x, attn_mask=mask)
        residual1 = x + self.dropout2(attened)
        norm1 = self.layer_norm2(residual1)
        feed_forward_out = self.feed_forward2(norm1)
        residual2 = norm1 + self.dropout2(feed_forward_out)
        norm2 = self.layer_norm22(residual2)
        norm1 = norm2.squeeze(1)
        out_transformer = self.fc2(norm2)
        out_transformer = self.dropout2(out_transformer)
        out_transformer = out_transformer.squeeze(dim=1)

        x_biLstm1 = x.to(torch.float32)
        _, (hidden_bilstm1, _) = self.bilstm2(x_biLstm1)
        final1 = torch.cat((out_cnn1, out_transformer, hidden_bilstm1[-2], hidden_bilstm1[-1], ), dim=1)
        return out_cnn0, norm1, torch.cat((hidden_bilstm1[-2], hidden_bilstm1[-1]), dim=1),final1

    def forward(self, x1, x2, x3, mask=None):
        x1_cnn, x1_trans, x1_bilstm, x1_final = self.prott5(x1)
        x2_cnn, x2_trans, x2_bilstm, x2_final = self.esm(x2)
        x3_cnn, x3_trans, x3_bilstm, x3_final = self.esm(x3)
        final4 = torch.cat((x1_final, x2_final, x3_final), dim=1)
        out = self.fc_final1(final4)
        out = self.fc_final2(out)
        return x1_cnn, x1_trans, x1_bilstm,x2_cnn, x2_trans, x2_bilstm,x3_cnn, x3_trans, x3_bilstm,out