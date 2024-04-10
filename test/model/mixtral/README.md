This directory stores the tests for various components of the Mixtral 8x7B model. Specifically, there are:
1. MixtralAttentionBlock: `test/model/mixtral/test_mixtral_attention.py`
2. MixtralSparseMoeBlock: `test/model/mixtral/test_mixtral_sparse_moe.py`
3. MixtralRMSNorm: Same as Llama's RMSNorm. See `test/model/open_llama/test_rms_norm.py` instead.
4. MixtralDecoderLayer: `test/model/mixtral/test_mixtral_decoder_layer.py`

More over, we also add an E2E test of a `small` Mixtral 8x7B model. You can see how to combine multiple parallel strategies (including TP/SP, DP and ZeRO 2+) to train a simple Mixtral network, refer to `test/model/mixtral/test_mixtral.py` for detail.