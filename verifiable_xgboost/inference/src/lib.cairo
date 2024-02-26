use orion::operators::tensor::{Tensor, TensorTrait};
use orion::operators::tensor::{U32Tensor, I32Tensor, I8Tensor, FP8x23Tensor, FP16x16Tensor, FP32x32Tensor, BoolTensor};
use orion::numbers::{FP8x23, FP16x16, FP32x32};
use orion::operators::matrix::{MutMatrix, MutMatrixImpl};
use orion::operators::ml;


fn main() -> (Span<u32>, MutMatrix<FP16x16>) {

    let node_input = TensorTrait::new(
            array![1,6].span(),
            array![
                FP16x16 { mag: 61166, sign: false },
                FP16x16 { mag: 65536, sign: false },
                FP16x16 { mag: 65536, sign: false },
                FP16x16 { mag: 65536, sign: false },
                FP16x16 { mag: 65536, sign: false },
                FP16x16 { mag: 0, sign: false },
                ].span()
    );
        
    let atts = ml::TreeEnsembleAttributes {
        nodes_falsenodeids: array![4,3,0,0,0,4,3,0,0,0,2,0,4,0,6,0,8,0,0,2,0,4,0,0,4,3,0,0,0,4,3,0,0,0,2,0,4,0,6,0,8,0,0,2,0,4,0,0,4,3,0,0,0,4,3,0,0,0,2,0,4,0,6,0,8,0,0,2,0,4,0,0,4,3,0,0,0,4,3,0,0,0,2,0,4,0,6,0,8,0,0,2,0,4,0,0,4,3,0,0,0,4,3,0,0,0,2,0,4,0,6,0,8,0,0,2,0,4,0,0,4,3,0,0,0,4,3,0,0,0,2,0,4,0,6,0,8,0,0,2,0,4,0,0,4,3,0,0,0,4,3,0,0,0,2,0,4,0,6,0,8,0,0,2,0,4,0,0,4,3,0,0,0,4,3,0,0,0,2,0,6,5,0,0,0,2,0,6,5,0,0,0,4,3,0,0,0,4,3,0,0,0,2,0,4,0,6,0,8,0,0,2,0,4,0,0,2,0,0,4,3,0,0,0,2,0,4,0,6,0,8,0,0,2,0,6,5,0,0,0].span(), 
        nodes_featureids: array![3,1,0,0,0,2,0,0,0,0,0,0,2,0,1,0,3,0,0,1,0,3,0,0,3,1,0,0,0,2,0,0,0,0,0,0,2,0,1,0,3,0,0,1,0,3,0,0,3,1,0,0,0,2,0,0,0,0,0,0,2,0,1,0,3,0,0,1,0,3,0,0,3,1,0,0,0,2,0,0,0,0,0,0,2,0,1,0,3,0,0,1,0,3,0,0,3,1,0,0,0,2,0,0,0,0,0,0,2,0,1,0,3,0,0,1,0,3,0,0,3,1,0,0,0,2,0,0,0,0,0,0,2,0,1,0,3,0,0,1,0,3,0,0,3,1,0,0,0,2,0,0,0,0,0,0,2,0,1,0,3,0,0,1,0,3,0,0,3,0,0,0,0,2,0,0,0,0,0,0,5,0,0,0,0,1,0,5,0,0,0,0,3,1,0,0,0,2,0,0,0,0,0,0,0,0,3,0,1,0,0,1,0,3,0,0,3,0,0,0,4,0,0,0,2,0,0,0,3,0,1,0,0,1,0,5,0,0,0,0].span(),
        nodes_missing_value_tracks_true: array![0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0].span(), 
        nodes_modes: array![ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::BRANCH_LT,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF,ml::NODE_MODES::LEAF].span(), 
        nodes_nodeids: array![0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,5,6,7,8,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,5,6,7,8,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,5,6,7,8,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,5,6,7,8,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,5,6,7,8,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,5,6,7,8,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,5,6,7,8,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,5,6,0,1,2,3,4,5,6,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,5,6,7,8,0,1,2,3,4,0,1,2,0,1,2,3,4,0,1,2,3,4,5,6,7,8,0,1,2,3,4,5,6].span(), 
        nodes_treeids: array![0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5,6,6,6,6,6,6,6,6,6,7,7,7,7,7,8,8,8,8,8,9,9,9,9,9,10,10,10,10,10,10,10,10,10,11,11,11,11,11,12,12,12,12,12,13,13,13,13,13,14,14,14,14,14,14,14,14,14,15,15,15,15,15,16,16,16,16,16,17,17,17,17,17,18,18,18,18,18,18,18,18,18,19,19,19,19,19,20,20,20,20,20,21,21,21,21,21,22,22,22,22,22,22,22,22,22,23,23,23,23,23,24,24,24,24,24,25,25,25,25,25,26,26,26,26,26,26,26,26,26,27,27,27,27,27,28,28,28,28,28,29,29,29,29,29,30,30,30,30,30,30,30,31,31,31,31,31,31,31,32,32,32,32,32,33,33,33,33,33,34,34,34,34,34,34,34,34,34,35,35,35,35,35,36,36,36,37,37,37,37,37,38,38,38,38,38,38,38,38,38,39,39,39,39,39,39,39].span(), 
        nodes_truenodeids: array![1,2,0,0,0,1,2,0,0,0,1,0,3,0,5,0,7,0,0,1,0,3,0,0,1,2,0,0,0,1,2,0,0,0,1,0,3,0,5,0,7,0,0,1,0,3,0,0,1,2,0,0,0,1,2,0,0,0,1,0,3,0,5,0,7,0,0,1,0,3,0,0,1,2,0,0,0,1,2,0,0,0,1,0,3,0,5,0,7,0,0,1,0,3,0,0,1,2,0,0,0,1,2,0,0,0,1,0,3,0,5,0,7,0,0,1,0,3,0,0,1,2,0,0,0,1,2,0,0,0,1,0,3,0,5,0,7,0,0,1,0,3,0,0,1,2,0,0,0,1,2,0,0,0,1,0,3,0,5,0,7,0,0,1,0,3,0,0,1,2,0,0,0,1,2,0,0,0,1,0,3,4,0,0,0,1,0,3,4,0,0,0,1,2,0,0,0,1,2,0,0,0,1,0,3,0,5,0,7,0,0,1,0,3,0,0,1,0,0,1,2,0,0,0,1,0,3,0,5,0,7,0,0,1,0,3,4,0,0,0].span(), 
        nodes_values: array![FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 27306, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 27306, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 27306, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 27306, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 27306, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 27306, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 27306, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 27306, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 27306, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 27306, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 27306, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 27306, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 27306, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 27306, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 26214, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 26214, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 27306, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 56797, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 56797, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 26214, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 27306, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 49152, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 27306, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 30583, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 65536, sign: false },FP16x16 { mag: 56797, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false },FP16x16 { mag: 0, sign: false }].span()
    };
     

        let mut root_index: Felt252Dict<usize> = Default::default();
    root_index.insert(0, 0);
    root_index.insert(1, 5);
    root_index.insert(2, 10);
    root_index.insert(3, 19);
    root_index.insert(4, 24);
    root_index.insert(5, 29);
    root_index.insert(6, 34);
    root_index.insert(7, 43);
    root_index.insert(8, 48);
    root_index.insert(9, 53);
    root_index.insert(10, 58);
    root_index.insert(11, 67);
    root_index.insert(12, 72);
    root_index.insert(13, 77);
    root_index.insert(14, 82);
    root_index.insert(15, 91);
    root_index.insert(16, 96);
    root_index.insert(17, 101);
    root_index.insert(18, 106);
    root_index.insert(19, 115);
    root_index.insert(20, 120);
    root_index.insert(21, 125);
    root_index.insert(22, 130);
    root_index.insert(23, 139);
    root_index.insert(24, 144);
    root_index.insert(25, 149);
    root_index.insert(26, 154);
    root_index.insert(27, 163);
    root_index.insert(28, 168);
    root_index.insert(29, 173);
    root_index.insert(30, 178);
    root_index.insert(31, 185);
    root_index.insert(32, 192);
    root_index.insert(33, 197);
    root_index.insert(34, 202);
    root_index.insert(35, 211);
    root_index.insert(36, 216);
    root_index.insert(37, 219);
    root_index.insert(38, 224);
    root_index.insert(39, 233); 

        let mut node_index: Felt252Dict<usize> = Default::default();
    node_index.insert(0, 0);
    node_index.insert(1, 1);
    node_index.insert(2, 2);
    node_index.insert(3, 3);
    node_index.insert(4, 4);
    node_index.insert(1, 5);
    node_index.insert(0, 6);
    node_index.insert(3, 7);
    node_index.insert(2, 8);
    node_index.insert(5, 9);
    node_index.insert(2, 10);
    node_index.insert(3, 11);
    node_index.insert(0, 12);
    node_index.insert(1, 13);
    node_index.insert(6, 14);
    node_index.insert(7, 15);
    node_index.insert(4, 16);
    node_index.insert(5, 17);
    node_index.insert(10, 18);
    node_index.insert(3, 19);
    node_index.insert(2, 20);
    node_index.insert(1, 21);
    node_index.insert(0, 22);
    node_index.insert(7, 23);
    node_index.insert(4, 24);
    node_index.insert(5, 25);
    node_index.insert(6, 26);
    node_index.insert(7, 27);
    node_index.insert(0, 28);
    node_index.insert(5, 29);
    node_index.insert(4, 30);
    node_index.insert(7, 31);
    node_index.insert(6, 32);
    node_index.insert(1, 33);
    node_index.insert(6, 34);
    node_index.insert(7, 35);
    node_index.insert(4, 36);
    node_index.insert(5, 37);
    node_index.insert(2, 38);
    node_index.insert(3, 39);
    node_index.insert(0, 40);
    node_index.insert(1, 41);
    node_index.insert(14, 42);
    node_index.insert(7, 43);
    node_index.insert(6, 44);
    node_index.insert(5, 45);
    node_index.insert(4, 46);
    node_index.insert(3, 47);
    node_index.insert(8, 48);
    node_index.insert(9, 49);
    node_index.insert(10, 50);
    node_index.insert(11, 51);
    node_index.insert(12, 52);
    node_index.insert(9, 53);
    node_index.insert(8, 54);
    node_index.insert(11, 55);
    node_index.insert(10, 56);
    node_index.insert(13, 57);
    node_index.insert(10, 58);
    node_index.insert(11, 59);
    node_index.insert(8, 60);
    node_index.insert(9, 61);
    node_index.insert(14, 62);
    node_index.insert(15, 63);
    node_index.insert(12, 64);
    node_index.insert(13, 65);
    node_index.insert(2, 66);
    node_index.insert(11, 67);
    node_index.insert(10, 68);
    node_index.insert(9, 69);
    node_index.insert(8, 70);
    node_index.insert(15, 71);
    node_index.insert(12, 72);
    node_index.insert(13, 73);
    node_index.insert(14, 74);
    node_index.insert(15, 75);
    node_index.insert(8, 76);
    node_index.insert(13, 77);
    node_index.insert(12, 78);
    node_index.insert(15, 79);
    node_index.insert(14, 80);
    node_index.insert(9, 81);
    node_index.insert(14, 82);
    node_index.insert(15, 83);
    node_index.insert(12, 84);
    node_index.insert(13, 85);
    node_index.insert(10, 86);
    node_index.insert(11, 87);
    node_index.insert(8, 88);
    node_index.insert(9, 89);
    node_index.insert(6, 90);
    node_index.insert(15, 91);
    node_index.insert(14, 92);
    node_index.insert(13, 93);
    node_index.insert(12, 94);
    node_index.insert(11, 95);
    node_index.insert(16, 96);
    node_index.insert(17, 97);
    node_index.insert(18, 98);
    node_index.insert(19, 99);
    node_index.insert(20, 100);
    node_index.insert(17, 101);
    node_index.insert(16, 102);
    node_index.insert(19, 103);
    node_index.insert(18, 104);
    node_index.insert(21, 105);
    node_index.insert(18, 106);
    node_index.insert(19, 107);
    node_index.insert(16, 108);
    node_index.insert(17, 109);
    node_index.insert(22, 110);
    node_index.insert(23, 111);
    node_index.insert(20, 112);
    node_index.insert(21, 113);
    node_index.insert(26, 114);
    node_index.insert(19, 115);
    node_index.insert(18, 116);
    node_index.insert(17, 117);
    node_index.insert(16, 118);
    node_index.insert(23, 119);
    node_index.insert(20, 120);
    node_index.insert(21, 121);
    node_index.insert(22, 122);
    node_index.insert(23, 123);
    node_index.insert(16, 124);
    node_index.insert(21, 125);
    node_index.insert(20, 126);
    node_index.insert(23, 127);
    node_index.insert(22, 128);
    node_index.insert(17, 129);
    node_index.insert(22, 130);
    node_index.insert(23, 131);
    node_index.insert(20, 132);
    node_index.insert(21, 133);
    node_index.insert(18, 134);
    node_index.insert(19, 135);
    node_index.insert(16, 136);
    node_index.insert(17, 137);
    node_index.insert(30, 138);
    node_index.insert(23, 139);
    node_index.insert(22, 140);
    node_index.insert(21, 141);
    node_index.insert(20, 142);
    node_index.insert(19, 143);
    node_index.insert(24, 144);
    node_index.insert(25, 145);
    node_index.insert(26, 146);
    node_index.insert(27, 147);
    node_index.insert(28, 148);
    node_index.insert(25, 149);
    node_index.insert(24, 150);
    node_index.insert(27, 151);
    node_index.insert(26, 152);
    node_index.insert(29, 153);
    node_index.insert(26, 154);
    node_index.insert(27, 155);
    node_index.insert(24, 156);
    node_index.insert(25, 157);
    node_index.insert(30, 158);
    node_index.insert(31, 159);
    node_index.insert(28, 160);
    node_index.insert(29, 161);
    node_index.insert(18, 162);
    node_index.insert(27, 163);
    node_index.insert(26, 164);
    node_index.insert(25, 165);
    node_index.insert(24, 166);
    node_index.insert(31, 167);
    node_index.insert(28, 168);
    node_index.insert(29, 169);
    node_index.insert(30, 170);
    node_index.insert(31, 171);
    node_index.insert(24, 172);
    node_index.insert(29, 173);
    node_index.insert(28, 174);
    node_index.insert(31, 175);
    node_index.insert(30, 176);
    node_index.insert(25, 177);
    node_index.insert(30, 178);
    node_index.insert(31, 179);
    node_index.insert(28, 180);
    node_index.insert(29, 181);
    node_index.insert(26, 182);
    node_index.insert(27, 183);
    node_index.insert(24, 184);
    node_index.insert(31, 185);
    node_index.insert(30, 186);
    node_index.insert(29, 187);
    node_index.insert(28, 188);
    node_index.insert(27, 189);
    node_index.insert(26, 190);
    node_index.insert(25, 191);
    node_index.insert(32, 192);
    node_index.insert(33, 193);
    node_index.insert(34, 194);
    node_index.insert(35, 195);
    node_index.insert(36, 196);
    node_index.insert(33, 197);
    node_index.insert(32, 198);
    node_index.insert(35, 199);
    node_index.insert(34, 200);
    node_index.insert(37, 201);
    node_index.insert(34, 202);
    node_index.insert(35, 203);
    node_index.insert(32, 204);
    node_index.insert(33, 205);
    node_index.insert(38, 206);
    node_index.insert(39, 207);
    node_index.insert(36, 208);
    node_index.insert(37, 209);
    node_index.insert(42, 210);
    node_index.insert(35, 211);
    node_index.insert(34, 212);
    node_index.insert(33, 213);
    node_index.insert(32, 214);
    node_index.insert(39, 215);
    node_index.insert(36, 216);
    node_index.insert(37, 217);
    node_index.insert(38, 218);
    node_index.insert(37, 219);
    node_index.insert(36, 220);
    node_index.insert(39, 221);
    node_index.insert(38, 222);
    node_index.insert(33, 223);
    node_index.insert(38, 224);
    node_index.insert(39, 225);
    node_index.insert(36, 226);
    node_index.insert(37, 227);
    node_index.insert(34, 228);
    node_index.insert(35, 229);
    node_index.insert(32, 230);
    node_index.insert(33, 231);
    node_index.insert(46, 232);
    node_index.insert(39, 233);
    node_index.insert(38, 234);
    node_index.insert(37, 235);
    node_index.insert(36, 236);
    node_index.insert(35, 237);
    node_index.insert(34, 238);
    node_index.insert(33, 239); 

        
    let mut ensemble = ml::TreeEnsemble {
        atts: atts, 
        tree_ids: array![0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39].span(), 
        root_index: root_index, 
        node_index: node_index
    };
     

        
    let mut classifier = ml::TreeEnsembleClassifier {
        ensemble: ensemble, 
        class_ids: array![0,0,0,1,1,1,2,2,2,2,2,3,3,3,0,0,0,1,1,1,2,2,2,2,2,3,3,3,0,0,0,1,1,1,2,2,2,2,2,3,3,3,0,0,0,1,1,1,2,2,2,2,2,3,3,3,0,0,0,1,1,1,2,2,2,2,2,3,3,3,0,0,0,1,1,1,2,2,2,2,2,3,3,3,0,0,0,1,1,1,2,2,2,2,2,3,3,3,0,0,0,1,1,1,2,2,2,2,3,3,3,3,0,0,0,1,1,1,2,2,2,2,2,3,3,3,0,0,1,1,1,2,2,2,2,2,3,3,3,3].span(), 
        class_nodeids: array![2,3,4,2,3,4,1,3,5,7,8,1,3,4,2,3,4,2,3,4,1,3,5,7,8,1,3,4,2,3,4,2,3,4,1,3,5,7,8,1,3,4,2,3,4,2,3,4,1,3,5,7,8,1,3,4,2,3,4,2,3,4,1,3,5,7,8,1,3,4,2,3,4,2,3,4,1,3,5,7,8,1,3,4,2,3,4,2,3,4,1,3,5,7,8,1,3,4,2,3,4,2,3,4,1,4,5,6,1,4,5,6,2,3,4,2,3,4,1,3,5,7,8,1,3,4,1,2,2,3,4,1,3,5,7,8,1,4,5,6].span(), 
        class_treeids: array![0,0,0,1,1,1,2,2,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,6,6,7,7,7,8,8,8,9,9,9,10,10,10,10,10,11,11,11,12,12,12,13,13,13,14,14,14,14,14,15,15,15,16,16,16,17,17,17,18,18,18,18,18,19,19,19,20,20,20,21,21,21,22,22,22,22,22,23,23,23,24,24,24,25,25,25,26,26,26,26,26,27,27,27,28,28,28,29,29,29,30,30,30,30,31,31,31,31,32,32,32,33,33,33,34,34,34,34,34,35,35,35,36,36,37,37,37,38,38,38,38,38,39,39,39,39].span(), 
        class_weights: array![FP16x16 { mag: 35389, sign: false },FP16x16 { mag: 9830, sign: true },FP16x16 { mag: 12582, sign: true },FP16x16 { mag: 36296, sign: false },FP16x16 { mag: 10111, sign: true },FP16x16 { mag: 12501, sign: true },FP16x16 { mag: 12403, sign: true },FP16x16 { mag: 10111, sign: true },FP16x16 { mag: 33704, sign: false },FP16x16 { mag: 29491, sign: false },FP16x16 { mag: 11234, sign: true },FP16x16 { mag: 12639, sign: true },FP16x16 { mag: 9830, sign: true },FP16x16 { mag: 33704, sign: false },FP16x16 { mag: 22238, sign: false },FP16x16 { mag: 9020, sign: true },FP16x16 { mag: 11667, sign: true },FP16x16 { mag: 22384, sign: false },FP16x16 { mag: 9149, sign: true },FP16x16 { mag: 11603, sign: true },FP16x16 { mag: 11463, sign: true },FP16x16 { mag: 9149, sign: true },FP16x16 { mag: 21582, sign: false },FP16x16 { mag: 20395, sign: false },FP16x16 { mag: 10322, sign: true },FP16x16 { mag: 11715, sign: true },FP16x16 { mag: 9020, sign: true },FP16x16 { mag: 21671, sign: false },FP16x16 { mag: 17036, sign: false },FP16x16 { mag: 8274, sign: true },FP16x16 { mag: 10972, sign: true },FP16x16 { mag: 17086, sign: false },FP16x16 { mag: 8333, sign: true },FP16x16 { mag: 10904, sign: true },FP16x16 { mag: 10742, sign: true },FP16x16 { mag: 8333, sign: true },FP16x16 { mag: 16522, sign: false },FP16x16 { mag: 15917, sign: false },FP16x16 { mag: 9544, sign: true },FP16x16 { mag: 11022, sign: true },FP16x16 { mag: 8274, sign: true },FP16x16 { mag: 16625, sign: false },FP16x16 { mag: 14248, sign: false },FP16x16 { mag: 7585, sign: true },FP16x16 { mag: 10409, sign: true },FP16x16 { mag: 14291, sign: false },FP16x16 { mag: 7595, sign: true },FP16x16 { mag: 10331, sign: true },FP16x16 { mag: 10146, sign: true },FP16x16 { mag: 7595, sign: true },FP16x16 { mag: 13747, sign: false },FP16x16 { mag: 13276, sign: false },FP16x16 { mag: 8848, sign: true },FP16x16 { mag: 10468, sign: true },FP16x16 { mag: 7585, sign: true },FP16x16 { mag: 13854, sign: false },FP16x16 { mag: 12518, sign: false },FP16x16 { mag: 6944, sign: true },FP16x16 { mag: 9929, sign: true },FP16x16 { mag: 12577, sign: false },FP16x16 { mag: 6912, sign: true },FP16x16 { mag: 9837, sign: true },FP16x16 { mag: 9624, sign: true },FP16x16 { mag: 6912, sign: true },FP16x16 { mag: 11990, sign: false },FP16x16 { mag: 11528, sign: false },FP16x16 { mag: 8204, sign: true },FP16x16 { mag: 9999, sign: true },FP16x16 { mag: 6944, sign: true },FP16x16 { mag: 12099, sign: false },FP16x16 { mag: 11330, sign: false },FP16x16 { mag: 6344, sign: true },FP16x16 { mag: 9494, sign: true },FP16x16 { mag: 11414, sign: false },FP16x16 { mag: 6271, sign: true },FP16x16 { mag: 9387, sign: true },FP16x16 { mag: 9139, sign: true },FP16x16 { mag: 6271, sign: true },FP16x16 { mag: 10755, sign: false },FP16x16 { mag: 10266, sign: false },FP16x16 { mag: 7591, sign: true },FP16x16 { mag: 9578, sign: true },FP16x16 { mag: 6344, sign: true },FP16x16 { mag: 10870, sign: false },FP16x16 { mag: 10446, sign: false },FP16x16 { mag: 5784, sign: true },FP16x16 { mag: 9079, sign: true },FP16x16 { mag: 10556, sign: false },FP16x16 { mag: 5668, sign: true },FP16x16 { mag: 8955, sign: true },FP16x16 { mag: 8666, sign: true },FP16x16 { mag: 5668, sign: true },FP16x16 { mag: 9808, sign: false },FP16x16 { mag: 9290, sign: false },FP16x16 { mag: 7000, sign: true },FP16x16 { mag: 9178, sign: true },FP16x16 { mag: 5784, sign: true },FP16x16 { mag: 9933, sign: false },FP16x16 { mag: 8724, sign: false },FP16x16 { mag: 3863, sign: false },FP16x16 { mag: 8663, sign: true },FP16x16 { mag: 9701, sign: false },FP16x16 { mag: 3, sign: true },FP16x16 { mag: 8525, sign: true },FP16x16 { mag: 8188, sign: true },FP16x16 { mag: 3224, sign: false },FP16x16 { mag: 6, sign: true },FP16x16 { mag: 6216, sign: false },FP16x16 { mag: 8778, sign: true },FP16x16 { mag: 815, sign: false },FP16x16 { mag: 3922, sign: false },FP16x16 { mag: 7626, sign: false },FP16x16 { mag: 9321, sign: false },FP16x16 { mag: 5361, sign: true },FP16x16 { mag: 8294, sign: true },FP16x16 { mag: 9098, sign: false },FP16x16 { mag: 234, sign: true },FP16x16 { mag: 8161, sign: true },FP16x16 { mag: 7735, sign: true },FP16x16 { mag: 6824, sign: false },FP16x16 { mag: 4283, sign: false },FP16x16 { mag: 6824, sign: false },FP16x16 { mag: 6420, sign: true },FP16x16 { mag: 8415, sign: true },FP16x16 { mag: 5281, sign: true },FP16x16 { mag: 8814, sign: false },FP16x16 { mag: 5960, sign: false },FP16x16 { mag: 7884, sign: true },FP16x16 { mag: 1622, sign: false },FP16x16 { mag: 8366, sign: false },FP16x16 { mag: 7801, sign: true },FP16x16 { mag: 7390, sign: true },FP16x16 { mag: 2135, sign: true },FP16x16 { mag: 7922, sign: false },FP16x16 { mag: 7578, sign: false },FP16x16 { mag: 5875, sign: true },FP16x16 { mag: 8012, sign: true },FP16x16 { mag: 350, sign: false },FP16x16 { mag: 2798, sign: false },FP16x16 { mag: 6477, sign: false }].span(), 
        classlabels: array![0,1,2,3].span(), 
        base_values: Option::Some(array![FP16x16 { mag: 32768, sign: false },FP16x16 { mag: 32768, sign: false },FP16x16 { mag: 32768, sign: false },FP16x16 { mag: 32768, sign: false }].span()), 
        post_transform: ml::POST_TRANSFORM::SOFTMAX
    };
     

        ml::TreeEnsembleClassifierTrait::predict(ref classifier, node_input)
        

    }