from dgl.nn.pytorch import GATConv
import dgl.function as fn
from dgl.base import DGLError
from dgl.nn.functional import edge_softmax


class CustomGATConv(GATConv):
    def forward(self, graph, feat, edge_weight=None, get_attention=False):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "This is harmful for some applications, "
                        "causing silent performance regression. "
                        "Adding self-loop on the input graph by "
                        "calling `g = dgl.add_self_loop(g)` will resolve "
                        "the issue. Setting ``allow_zero_in_degree`` "
                        "to be `True` when constructing this module will "
                        "suppress the check and let the code run."
                    )

            # Calculate feature projection
            if isinstance(feat, tuple):
                src_prefix_shape = feat[0].shape[:-1]
                dst_prefix_shape = feat[1].shape[:-1]
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, "fc_src"):
                    feat_src = self.fc(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats
                    )
                    feat_dst = self.fc(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats
                    )
                else:
                    feat_src = self.fc_src(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats
                    )
                    feat_dst = self.fc_dst(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats
                    )
            else:
                src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    *src_prefix_shape, self._num_heads, self._out_feats
                )
                if graph.is_block:
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]
                    h_dst = h_dst[: graph.number_of_dst_nodes()]
                    dst_prefix_shape = (
                        graph.number_of_dst_nodes(),
                    ) + dst_prefix_shape[1:]

            # Attention mechanism
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({"ft": feat_src, "el": el})
            graph.dstdata.update({"er": er})
            graph.apply_edges(fn.u_add_v("el", "er", "e"))
            e = self.leaky_relu(graph.edata.pop("e"))
            attention_scores = edge_softmax(graph, e)

            # 设置 attention_scores 的 requires_grad 属性
            attention_scores = attention_scores.requires_grad_(True)

            # Make attention_scores retain gradient information
            attention_scores.retain_grad()
            graph.edata["a"] = self.attn_drop(attention_scores)

            # 检查 attention_scores 是否参与了计算图
            # print(f"Attention scores tensor: {attention_scores}, requires_grad={attention_scores.requires_grad}")

            if edge_weight is not None:
                graph.edata["a"] = graph.edata["a"] * edge_weight.tile(
                    1, self._num_heads, 1
                ).transpose(0, 2)

            graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            rst = graph.dstdata["ft"]

            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(
                    *dst_prefix_shape, -1, self._out_feats
                )
                rst = rst + resval

            if self.has_explicit_bias:
                rst = rst + self.bias.view(
                    *((1,) * len(dst_prefix_shape)),
                    self._num_heads,
                    self._out_feats
                )

            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, attention_scores
            else:
                return rst
