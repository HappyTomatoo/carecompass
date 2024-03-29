use alexandria_data_structures::vec::VecTrait;
use orion::operators::tensor::{Tensor, TensorTrait};
use orion::operators::tensor::{U32Tensor, I32Tensor, I8Tensor, FP8x23Tensor, FP16x16Tensor, FP32x32Tensor, BoolTensor};
use orion::numbers::{FixedTrait, FP16x16};
use orion::operators::matrix::{MutMatrix, MutMatrixImpl};
use orion::operators::ml;
use orion::operators::ml::tree_ensemble::core::{NODE_MODES, TreeEnsembleAttributes, TreeEnsemble};
use orion::operators::ml::tree_ensemble::tree_ensemble_classifier::{TreeEnsembleClassifier, POST_TRANSFORM, TreeEnsembleClassifierTrait};


// fn main(node_input: Tensor<FP16x16>) -> (Span<u32>, Tensor<FP16x16>) {
fn main() -> (Span<u32>, Tensor<FP16x16>) {

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
    // output 3
        
    let class_ids: Span<usize> = array![0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3].span();
    let class_nodeids: Span<usize> = array![2, 3, 4, 2, 3, 4, 1, 3, 5, 7, 8, 1, 3, 4, 2, 3, 4, 2, 3, 4, 1, 3, 5, 7, 8, 1, 3, 4, 2, 3, 4, 2, 3, 4, 1, 3, 5, 7, 8, 1, 3, 4, 2, 3, 4, 2, 3, 4, 1, 3, 5, 7, 8, 1, 3, 4, 2, 3, 4, 2, 3, 4, 1, 3, 5, 7, 8, 1, 3, 4, 2, 3, 4, 2, 3, 4, 1, 3, 5, 7, 8, 1, 3, 4, 2, 3, 4, 2, 3, 4, 1, 3, 4, 1, 3, 4, 2, 3, 4, 2, 3, 4, 1, 3, 4, 1, 3, 4, 2, 3, 4, 2, 3, 4, 1, 3, 5, 7, 8, 1, 3, 4, 2, 3, 4, 2, 3, 4, 1, 3, 5, 7, 8, 1, 4, 5, 6].span();
    let class_treeids: Span<usize> = array![0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10, 10, 10, 11, 11, 11, 12, 12, 12, 13, 13, 13, 14, 14, 14, 14, 14, 15, 15, 15, 16, 16, 16, 17, 17, 17, 18, 18, 18, 18, 18, 19, 19, 19, 20, 20, 20, 21, 21, 21, 22, 22, 22, 22, 22, 23, 23, 23, 24, 24, 24, 25, 25, 25, 26, 26, 26, 27, 27, 27, 28, 28, 28, 29, 29, 29, 30, 30, 30, 31, 31, 31, 32, 32, 32, 33, 33, 33, 34, 34, 34, 34, 34, 35, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38, 38, 38, 38, 39, 39, 39, 39].span();
    let class_weights: Span<FP16x16> = array![FP16x16 { mag: 34695, sign: false }, FP16x16 { mag: 10347, sign: true }, FP16x16 { mag: 12598, sign: true }, FP16x16 { mag: 36678, sign: false }, FP16x16 { mag: 9491, sign: true }, FP16x16 { mag: 12467, sign: true }, FP16x16 { mag: 12443, sign: true }, FP16x16 { mag: 9491, sign: true }, FP16x16 { mag: 33989, sign: false }, FP16x16 { mag: 31043, sign: false }, FP16x16 { mag: 10724, sign: true }, FP16x16 { mag: 12651, sign: true }, FP16x16 { mag: 10347, sign: true }, FP16x16 { mag: 32172, sign: false }, FP16x16 { mag: 22044, sign: false }, FP16x16 { mag: 9486, sign: true }, FP16x16 { mag: 11680, sign: true }, FP16x16 { mag: 22489, sign: false }, FP16x16 { mag: 8565, sign: true }, FP16x16 { mag: 11575, sign: true }, FP16x16 { mag: 11502, sign: true }, FP16x16 { mag: 8565, sign: true }, FP16x16 { mag: 21675, sign: false }, FP16x16 { mag: 20936, sign: false }, FP16x16 { mag: 9854, sign: true }, FP16x16 { mag: 11726, sign: true }, FP16x16 { mag: 9486, sign: true }, FP16x16 { mag: 21184, sign: false }, FP16x16 { mag: 16911, sign: false }, FP16x16 { mag: 8716, sign: true }, FP16x16 { mag: 10986, sign: true }, FP16x16 { mag: 17160, sign: false }, FP16x16 { mag: 7766, sign: true }, FP16x16 { mag: 10875, sign: true }, FP16x16 { mag: 10785, sign: true }, FP16x16 { mag: 7766, sign: true }, FP16x16 { mag: 16586, sign: false }, FP16x16 { mag: 16236, sign: false }, FP16x16 { mag: 9083, sign: true }, FP16x16 { mag: 11035, sign: true }, FP16x16 { mag: 8716, sign: true }, FP16x16 { mag: 16314, sign: false }, FP16x16 { mag: 14130, sign: false }, FP16x16 { mag: 8011, sign: true }, FP16x16 { mag: 10426, sign: true }, FP16x16 { mag: 14364, sign: false }, FP16x16 { mag: 7042, sign: true }, FP16x16 { mag: 10298, sign: true }, FP16x16 { mag: 10195, sign: true }, FP16x16 { mag: 7042, sign: true }, FP16x16 { mag: 13807, sign: false }, FP16x16 { mag: 13536, sign: false }, FP16x16 { mag: 8381, sign: true }, FP16x16 { mag: 10482, sign: true }, FP16x16 { mag: 8011, sign: true }, FP16x16 { mag: 13574, sign: false }, FP16x16 { mag: 12390, sign: false }, FP16x16 { mag: 7356, sign: true }, FP16x16 { mag: 9948, sign: true }, FP16x16 { mag: 12659, sign: false }, FP16x16 { mag: 6375, sign: true }, FP16x16 { mag: 9798, sign: true }, FP16x16 { mag: 9680, sign: true }, FP16x16 { mag: 6375, sign: true }, FP16x16 { mag: 12053, sign: false }, FP16x16 { mag: 11777, sign: false }, FP16x16 { mag: 7728, sign: true }, FP16x16 { mag: 10016, sign: true }, FP16x16 { mag: 7356, sign: true }, FP16x16 { mag: 11810, sign: false }, FP16x16 { mag: 11185, sign: false }, FP16x16 { mag: 6740, sign: true }, FP16x16 { mag: 9517, sign: true }, FP16x16 { mag: 11510, sign: false }, FP16x16 { mag: 5755, sign: true }, FP16x16 { mag: 9342, sign: true }, FP16x16 { mag: 9204, sign: true }, FP16x16 { mag: 5755, sign: true }, FP16x16 { mag: 10824, sign: false }, FP16x16 { mag: 10521, sign: false }, FP16x16 { mag: 7110, sign: true }, FP16x16 { mag: 9599, sign: true }, FP16x16 { mag: 6740, sign: true }, FP16x16 { mag: 10556, sign: false }, FP16x16 { mag: 10279, sign: false }, FP16x16 { mag: 6159, sign: true }, FP16x16 { mag: 9106, sign: true }, FP16x16 { mag: 10553, sign: false }, FP16x16 { mag: 862, sign: false }, FP16x16 { mag: 8903, sign: true }, FP16x16 { mag: 8740, sign: true }, FP16x16 { mag: 3417, sign: false }, FP16x16 { mag: 7800, sign: false }, FP16x16 { mag: 9202, sign: true }, FP16x16 { mag: 6159, sign: true }, FP16x16 { mag: 9591, sign: false }, FP16x16 { mag: 9645, sign: false }, FP16x16 { mag: 5831, sign: true }, FP16x16 { mag: 8718, sign: true }, FP16x16 { mag: 9876, sign: false }, FP16x16 { mag: 764, sign: false }, FP16x16 { mag: 8508, sign: true }, FP16x16 { mag: 8294, sign: true }, FP16x16 { mag: 3031, sign: false }, FP16x16 { mag: 6892, sign: false }, FP16x16 { mag: 8827, sign: true }, FP16x16 { mag: 5831, sign: true }, FP16x16 { mag: 9005, sign: false }, FP16x16 { mag: 9103, sign: false }, FP16x16 { mag: 5522, sign: true }, FP16x16 { mag: 8324, sign: true }, FP16x16 { mag: 9295, sign: false }, FP16x16 { mag: 679, sign: false }, FP16x16 { mag: 8113, sign: true }, FP16x16 { mag: 7838, sign: true }, FP16x16 { mag: 7157, sign: false }, FP16x16 { mag: 5132, sign: false }, FP16x16 { mag: 6853, sign: false }, FP16x16 { mag: 6328, sign: true }, FP16x16 { mag: 8445, sign: true }, FP16x16 { mag: 5522, sign: true }, FP16x16 { mag: 8506, sign: false }, FP16x16 { mag: 7077, sign: false }, FP16x16 { mag: 2413, sign: false }, FP16x16 { mag: 7916, sign: true }, FP16x16 { mag: 2396, sign: false }, FP16x16 { mag: 8618, sign: false }, FP16x16 { mag: 7745, sign: true }, FP16x16 { mag: 7515, sign: true }, FP16x16 { mag: 1713, sign: true }, FP16x16 { mag: 8106, sign: false }, FP16x16 { mag: 7752, sign: false }, FP16x16 { mag: 5784, sign: true }, FP16x16 { mag: 8045, sign: true }, FP16x16 { mag: 2135, sign: false }, FP16x16 { mag: 198, sign: false }, FP16x16 { mag: 5671, sign: false }].span();
    let classlabels: Span<usize> = array![0, 1, 2, 3].span();
    let nodes_falsenodeids: Span<usize> = array![4, 3, 0, 0, 0, 4, 3, 0, 0, 0, 2, 0, 4, 0, 6, 0, 8, 0, 0, 2, 0, 4, 0, 0, 4, 3, 0, 0, 0, 4, 3, 0, 0, 0, 2, 0, 4, 0, 6, 0, 8, 0, 0, 2, 0, 4, 0, 0, 4, 3, 0, 0, 0, 4, 3, 0, 0, 0, 2, 0, 4, 0, 6, 0, 8, 0, 0, 2, 0, 4, 0, 0, 4, 3, 0, 0, 0, 4, 3, 0, 0, 0, 2, 0, 4, 0, 6, 0, 8, 0, 0, 2, 0, 4, 0, 0, 4, 3, 0, 0, 0, 4, 3, 0, 0, 0, 2, 0, 4, 0, 6, 0, 8, 0, 0, 2, 0, 4, 0, 0, 4, 3, 0, 0, 0, 4, 3, 0, 0, 0, 2, 0, 4, 0, 6, 0, 8, 0, 0, 2, 0, 4, 0, 0, 4, 3, 0, 0, 0, 4, 3, 0, 0, 0, 2, 0, 4, 0, 0, 2, 0, 4, 0, 0, 4, 3, 0, 0, 0, 4, 3, 0, 0, 0, 2, 0, 4, 0, 0, 2, 0, 4, 0, 0, 4, 3, 0, 0, 0, 4, 3, 0, 0, 0, 2, 0, 4, 0, 6, 0, 8, 0, 0, 2, 0, 4, 0, 0, 4, 3, 0, 0, 0, 4, 3, 0, 0, 0, 2, 0, 4, 0, 6, 0, 8, 0, 0, 2, 0, 6, 5, 0, 0, 0].span();
    let nodes_featureids: Span<usize> = array![3, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 3, 0, 0, 1, 0, 3, 0, 0, 3, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 3, 0, 0, 1, 0, 3, 0, 0, 3, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 3, 0, 0, 1, 0, 3, 0, 0, 3, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 3, 0, 0, 1, 0, 3, 0, 0, 3, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 3, 0, 0, 1, 0, 3, 0, 0, 3, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 3, 0, 0, 1, 0, 3, 0, 0, 3, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 5, 0, 0, 1, 0, 3, 0, 0, 3, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 5, 0, 0, 1, 0, 3, 0, 0, 3, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 1, 0, 0, 1, 0, 3, 0, 0, 3, 0, 0, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 3, 0, 1, 0, 0, 1, 0, 5, 0, 0, 0, 0].span();
    let nodes_missing_value_tracks_true: Span<usize> = array![1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0].span();
    let nodes_modes: Span<NODE_MODES> = array![NODE_MODES::BRANCH_LT, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::BRANCH_LT, NODE_MODES::BRANCH_LT, NODE_MODES::LEAF, NODE_MODES::LEAF, NODE_MODES::LEAF].span();
    let nodes_nodeids: Span<usize> = array![0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6].span();
    let nodes_treeids: Span<usize> = array![0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 26, 26, 26, 26, 26, 27, 27, 27, 27, 27, 28, 28, 28, 28, 28, 29, 29, 29, 29, 29, 30, 30, 30, 30, 30, 31, 31, 31, 31, 31, 32, 32, 32, 32, 32, 33, 33, 33, 33, 33, 34, 34, 34, 34, 34, 34, 34, 34, 34, 35, 35, 35, 35, 35, 36, 36, 36, 36, 36, 37, 37, 37, 37, 37, 38, 38, 38, 38, 38, 38, 38, 38, 38, 39, 39, 39, 39, 39, 39, 39].span();
    let nodes_truenodeids: Span<usize> = array![1, 2, 0, 0, 0, 1, 2, 0, 0, 0, 1, 0, 3, 0, 5, 0, 7, 0, 0, 1, 0, 3, 0, 0, 1, 2, 0, 0, 0, 1, 2, 0, 0, 0, 1, 0, 3, 0, 5, 0, 7, 0, 0, 1, 0, 3, 0, 0, 1, 2, 0, 0, 0, 1, 2, 0, 0, 0, 1, 0, 3, 0, 5, 0, 7, 0, 0, 1, 0, 3, 0, 0, 1, 2, 0, 0, 0, 1, 2, 0, 0, 0, 1, 0, 3, 0, 5, 0, 7, 0, 0, 1, 0, 3, 0, 0, 1, 2, 0, 0, 0, 1, 2, 0, 0, 0, 1, 0, 3, 0, 5, 0, 7, 0, 0, 1, 0, 3, 0, 0, 1, 2, 0, 0, 0, 1, 2, 0, 0, 0, 1, 0, 3, 0, 5, 0, 7, 0, 0, 1, 0, 3, 0, 0, 1, 2, 0, 0, 0, 1, 2, 0, 0, 0, 1, 0, 3, 0, 0, 1, 0, 3, 0, 0, 1, 2, 0, 0, 0, 1, 2, 0, 0, 0, 1, 0, 3, 0, 0, 1, 0, 3, 0, 0, 1, 2, 0, 0, 0, 1, 2, 0, 0, 0, 1, 0, 3, 0, 5, 0, 7, 0, 0, 1, 0, 3, 0, 0, 1, 2, 0, 0, 0, 1, 2, 0, 0, 0, 1, 0, 3, 0, 5, 0, 7, 0, 0, 1, 0, 3, 4, 0, 0, 0].span();
    let nodes_values: Span<FP16x16> = array![FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 37683, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 26760, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 37683, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 26760, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 37683, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 26760, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 37683, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 26760, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 37683, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 26760, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 37683, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 26760, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 25668, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 26760, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 25668, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 26760, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 25668, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 26760, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 47513, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 37137, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 26760, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 27852, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 55159, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 0, sign: false }].span();
    let base_values: Option<Span<FP16x16>> = Option::Some(array![FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 32768, sign: false }, FP16x16 { mag: 32768, sign: false }].span());
    let post_transform = POST_TRANSFORM::NONE;

    let tree_ids: Span<usize> = array![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39].span();
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
        root_index.insert(27, 159);
        root_index.insert(28, 164);
        root_index.insert(29, 169);
        root_index.insert(30, 174);
        root_index.insert(31, 179);
        root_index.insert(32, 184);
        root_index.insert(33, 189);
        root_index.insert(34, 194);
        root_index.insert(35, 203);
        root_index.insert(36, 208);
        root_index.insert(37, 213);
        root_index.insert(38, 218);
        root_index.insert(39, 227);
    let mut node_index: Felt252Dict<usize> = Default::default();
        node_index.insert(2089986280348253421170679821480865132823066470938446095505822317253594081284, 0);
        node_index.insert(2001140082530619239661729809084578298299223810202097622761632384561112390979, 1);
        node_index.insert(2592670241084192212354027440049085852792506518781954896144296316131790403900, 2);
        node_index.insert(2960591271376829378356567803618548672034867345123727178628869426548453833420, 3);
        node_index.insert(458933264452572171106695256465341160654132084710250671055261382009315664425, 4);
        node_index.insert(1089549915800264549621536909767699778745926517555586332772759280702396009108, 5);
        node_index.insert(1321142004022994845681377299801403567378503530250467610343381590909832171180, 6);
        node_index.insert(2592987851775965742543459319508348457290966253241455514226127639100457844774, 7);
        node_index.insert(2492755623019086109032247218615964389726368532160653497039005814484393419348, 8);
        node_index.insert(1323616023845704258113538348000047149470450086307731200728039607710316625916, 9);
        node_index.insert(1637368371864026355245122316446106576874611007407245016652355316950184561542, 10);
        node_index.insert(1207699383798263883125605407307435965808923448511613904826718551574712750645, 11);
        node_index.insert(1180550645873507273865212362837104046225859416703538577277065670066180087996, 12);
        node_index.insert(2472368796876167851123807933892689310864199328863324614163271115835436643256, 13);
        node_index.insert(2178161520066714737684323463974044933282313051386084149915030950231093462467, 14);
        node_index.insert(3173857952392006478115535740648337122591944728347430823115535852278920741329, 15);
        node_index.insert(3520866866675103377451014823284046708949378458998098483918774175589256898262, 16);
        node_index.insert(2177941723204159354958359305495360053392108189585761207084264896971635476845, 17);
        node_index.insert(2864394597864040652203681823189877789250870157950998029096571449554775379175, 18);
        node_index.insert(936823097115478672163131070534991867793647843312823827742596382032679996195, 19);
        node_index.insert(2908682032041418908903105681227249033483541201006723240850136728317167492227, 20);
        node_index.insert(576657123605396437968823113955952586959670965011232700393892413073919304299, 21);
        node_index.insert(1481896535808584724908081559736405324517902569915916348692801954386091870245, 22);
        node_index.insert(1078504723311822443900992338775481548059850561756203702548080974952533155775, 23);
        node_index.insert(469486474782544164430568959439120883383782181399389907385047779197726806430, 24);
        node_index.insert(3512521406437956009189089258567111789473785799907488469636118378769715425964, 25);
        node_index.insert(2556139128341700567231916301725351155453738182692001295135848448652163014397, 26);
        node_index.insert(3465043718230761803333064436233095750064993904445202331528878419074385927073, 27);
        node_index.insert(1484044891644535909221789528912343377988032083750620401102632587429863384003, 28);
        node_index.insert(2941083907689010536497253969578701440794094793277200004061830176674600429738, 29);
        node_index.insert(3515557115945123685249720924176918246289668839127088842764552624387741006658, 30);
        node_index.insert(1086529665842980980708131000626441572702884934518178946087373814825190924452, 31);
        node_index.insert(1577600695648543593850511011274446155472648910099879524512420959593356117933, 32);
        node_index.insert(369637003681923374235868884090690746347960427214527191195034764982555476019, 33);
        node_index.insert(2741690337285522037147443857948052150995543108052651970979313688522374979162, 34);
        node_index.insert(2650761223990278311391161549562769924185646525933398576366599729646470331982, 35);
        node_index.insert(2394106477146207452133044987332160354989937730594050309156799249985649638789, 36);
        node_index.insert(3221546555590725210494412430124213369510944559726456557849915592640050076870, 37);
        node_index.insert(2228295198404333119017490955293049364627235124737441027193226305830899856739, 38);
        node_index.insert(1500728334208813541595897459030132460053541518563429199208970697149622864812, 39);
        node_index.insert(3370962038708719437204954491911331590397946488012391643100034534008512344333, 40);
        node_index.insert(1316763448112859394321117487731514423037681243887874196205047698610445360084, 41);
        node_index.insert(2675592129551529371290790685310559984125358643115246876322043698639381749698, 42);
        node_index.insert(2258442912665439649622769515993460039756024697697714582745734598954638194578, 43);
        node_index.insert(1923650700608380821616803627552990459031020321822263486178231314533355655733, 44);
        node_index.insert(2986518118017342969503780420947472437589402926077471462482848861272477228312, 45);
        node_index.insert(1376854647019374078119330239267360951409489379782091263003138693189186942360, 46);
        node_index.insert(198240730684500640150208378390372236261578668680828477399089686184533568077, 47);
        node_index.insert(2743794648056839147566190792738700325779538550063233531691573479295033948774, 48);
        node_index.insert(2798268708043007987823290469469057887013592827991152425130480624165644530309, 49);
        node_index.insert(2656358495835759095543325181783754425097697418395968068218877742551382200798, 50);
        node_index.insert(1211642483429689718997906629094690895255021115365864106699680729506720772533, 51);
        node_index.insert(520077409223118195244629881107208371526219124916445328459593670003268088063, 52);
        node_index.insert(3149011590233272225803080114059308917528748800879621812239443987136907759492, 53);
        node_index.insert(1256253793249097778491586851395915710675714820964284887367516531463100834885, 54);
        node_index.insert(1367204028253064270272780398308197116331885826019160095546179574252301715156, 55);
        node_index.insert(35709238200630692715748986686945022954166018904577236782775928014549886022, 56);
        node_index.insert(1690135278907791721896273394500395617659804632461123206580263779149876929936, 57);
        node_index.insert(2466881358002133364822637278001945633159199669109451817445969730922553850042, 58);
        node_index.insert(1183542427150969392626018984893467106583918766494481729382281203507622641472, 59);
        node_index.insert(1431776777056506698604647341381142583651190852368505702998607062255574129691, 60);
        node_index.insert(972438690926792325118135862382080018232032520094840957911232540298409406549, 61);
        node_index.insert(991084547888087131868886386324316432359028694627723192745237264535932014843, 62);
        node_index.insert(1426554411334771946627122235132505593863675277058690422710188841592243493500, 63);
        node_index.insert(1363909600341144275090208595536779376858146627250961583203694662295241812700, 64);
        node_index.insert(2535427936407932761861381573383136997958868317407701167331468890664673428752, 65);
        node_index.insert(358201485054646638265237224658728272117621129993093986898837614087243093706, 66);
        node_index.insert(1602195742608144856779311879863141684990052756940086705696922586637104021594, 67);
        node_index.insert(1048567650354208837177239552642912548937251398366633889054635887815999325708, 68);
        node_index.insert(952677257945065829974687215393982028642635658856182966013273240312944498520, 69);
        node_index.insert(3129863851213724954715979467157273023250206055497389996487004326392731096500, 70);
        node_index.insert(443979287152462807449557087711420799339405906857860200472038467754978880122, 71);
        node_index.insert(108289613193197664273020636632286985109898695751875754699753665839093823971, 72);
        node_index.insert(3121766102057675359404485811098033715935569748843064794474925162873415592191, 73);
        node_index.insert(2842835726332689313526529973184323329623799418065470244629244349557488332087, 74);
        node_index.insert(3586889477532689702613742797299212047244535726089914347286398831811496165403, 75);
        node_index.insert(719156341629408405425801415951358931787136862349280099344980025288394333850, 76);
        node_index.insert(1744979027318172964422327404361273580944397535401008023582199876115845114710, 77);
        node_index.insert(3243570830066835454087140059665140442995908735608860300830191054712060175676, 78);
        node_index.insert(3380860025071115828135533364448960345664402938799669705180316161453747776812, 79);
        node_index.insert(289733928291422171570659757954000263796467129100516441407291220571896851292, 80);
        node_index.insert(1953138310648956024269190916206716324541670224299230904312462701388831706877, 81);
        node_index.insert(1398462719893551374966620069731053475358010744206910073704581388509079808738, 82);
        node_index.insert(2932104990710995491165499208787426296543942318468262181074380705763249912480, 83);
        node_index.insert(1352327956410336250118662269404377770784993255864929446536502728911484427472, 84);
        node_index.insert(512141981192653777354314890546016960676867395426731826410027531503616663631, 85);
        node_index.insert(3506000480399786945731503406271917240022044946679358289040065268208299861777, 86);
        node_index.insert(2585970149770899244718593537196553182368312374131478958649846430872123560833, 87);
        node_index.insert(90076454508904655614246376810970822494240542463792212186146380473817826531, 88);
        node_index.insert(193249753264562328399183129502906237619350521783475455954855166607739230500, 89);
        node_index.insert(2258440615058254029478359682082494591491725245284400402019654393900880176408, 90);
        node_index.insert(1901504716446065256269731867411614915820094303385453967240909894258876518659, 91);
        node_index.insert(2342634400607154708670819644735766454051191896102045087656554437413520673159, 92);
        node_index.insert(2042972120432632004804992194872886828188714797262971027898719585329973786140, 93);
        node_index.insert(2919773935475631199511104804254524516982927944377711070272850657768492620157, 94);
        node_index.insert(2912835668604137948136888887066489630113541334244871225219028605879334191553, 95);
        node_index.insert(1798359565014827938354315398572476100256405791288404013872250892856849691828, 96);
        node_index.insert(3497930283373730048819686947750591200849592984585603643342875353326392194382, 97);
        node_index.insert(49159210227923046234273721687186127789670623748290282669921094555490563152, 98);
        node_index.insert(1449794955473981148369276697877160239367821034081828379021420212251540147967, 99);
        node_index.insert(3278402006889156212396203321338750509201085337254121695424279758626531575672, 100);
        node_index.insert(2017341737295481246155952905134235557561671392775186148749240339082999851285, 101);
        node_index.insert(2675393266147713586540390711506539058245204064012332842480150188957678948642, 102);
        node_index.insert(1174320953419110275179679712994777767970657923950362192771715320325831893719, 103);
        node_index.insert(33937463903191888230815642502975961825698474039859721115366149885182728562, 104);
        node_index.insert(3230832381355607450741945124470920607805481360860820557308094751239351857915, 105);
        node_index.insert(2005609713613971083569730539855781060205500887155286737608240129703145008240, 106);
        node_index.insert(1938582584258338730735878877878677528935257985543012616039874508134968503229, 107);
        node_index.insert(602031794088793330349673093703182401607695223031859881472597886958206737986, 108);
        node_index.insert(708113740382739776120838691985660267219327467914879680089389175082768408583, 109);
        node_index.insert(2347135847807611223792315040677498937185946059353375555547873583067254130883, 110);
        node_index.insert(1774386306390375443742680226459919375176612801430177973884064175685557733612, 111);
        node_index.insert(1115300594142736732120560543954295704889179999955771152165051906517017059990, 112);
        node_index.insert(1649451397574140763143931996368646788189769327014873833752667708858973754414, 113);
        node_index.insert(2750619522583353481534177929087286674695871588574849146827928263148456278950, 114);
        node_index.insert(3448995911069818038857594699452213191125033055828900916127480888562343134855, 115);
        node_index.insert(694441606487233543357610438873534908734548082720884370387398420315477789828, 116);
        node_index.insert(597541471833237499619329235557116049144965796360920672434662066360950273594, 117);
        node_index.insert(943352487530717527458209262144828430502222574259572194276717313521948410631, 118);
        node_index.insert(1837807953466086232851139042153308474077075180401446290159520516521762626697, 119);
        node_index.insert(863102740359469064208350212255578989777360247304628303691047272311665540549, 120);
        node_index.insert(1546170469693331057685767647163210997435398791854125937885688522307012221348, 121);
        node_index.insert(1449275071102314837783566626980758998963737493418232585653094618837080840554, 122);
        node_index.insert(996844148196859586868358098184067175687403705923353538158078660698860072156, 123);
        node_index.insert(793811170627544117106156237060663692288360890443228274026891329652191595806, 124);
        node_index.insert(1485445020346165906255248681301905749937376961180871925086218332677909156603, 125);
        node_index.insert(3156606750941688466303304743424767721761739343108206863775009483231225943584, 126);
        node_index.insert(92194528276771605719434380931894269977870018710968393185001169123067894064, 127);
        node_index.insert(2172113288928946773858164888287407392511131650559030529477693054219775323910, 128);
        node_index.insert(3064570890882091718944639785868277551066812816435506020353196104341690813903, 129);
        node_index.insert(900843776628869095842392032788189492233078812152813397832903878589633056377, 130);
        node_index.insert(2977730235565230139359688263419835536347671334868196477504334799643754355042, 131);
        node_index.insert(886036718756895177446882047945715748168696112114609507233877991791793965448, 132);
        node_index.insert(690633889716624434677672939175461712549684187607704468749467256553512926937, 133);
        node_index.insert(1007259422194864096467321328839734794999546610506683992728493345746539509072, 134);
        node_index.insert(2716056574597937872135657204617373288438106751903272696180443420369299694657, 135);
        node_index.insert(1185605736903872453746521051157493242433213565302871389306563559800991076016, 136);
        node_index.insert(378633997989415448323141990415083426003014918622360185720000299807622550247, 137);
        node_index.insert(280605053224506425198846009863070141837796755229414335051764989422618345874, 138);
        node_index.insert(3328320341141598096676572331686026435109492665031365053408653079689078671154, 139);
        node_index.insert(490421097117586408026586959556259109407325741858916058020034654364163085567, 140);
        node_index.insert(658703487938413674041539468518400908567835118348818752273965721284121653677, 141);
        node_index.insert(2723738314168995601320820958849252705861331688136261156861033214745374537727, 142);
        node_index.insert(343398506254743217551791302726931049585917637325641873360484577406648361116, 143);
        node_index.insert(3040393303064235409123029443843570136174868409519784248230541132797440323641, 144);
        node_index.insert(2350643951558905987951001431053207051197170256330463295783682739055185682885, 145);
        node_index.insert(2458141537053938027535006939534457636562189613276290899195398843748329842469, 146);
        node_index.insert(247953353508102457574889489399939633660179375824418843945501925222279777041, 147);
        node_index.insert(1164663877045092838861776945058381693262557717898631531735680962905052974232, 148);
        node_index.insert(2372264378636910324181215552780751393495517136504008526087815157936299073206, 149);
        node_index.insert(3594334860858900419027719494991615202081221749774690557072136191797036841030, 150);
        node_index.insert(561761507351273254574299626060599751429571869916050027217942782066019358030, 151);
        node_index.insert(2649958511896619304831126612860009755478676680611288948021339846080563774028, 152);
        node_index.insert(16170664024491513845994591584180975943645947068223432188175979470508390034, 153);
        node_index.insert(2762899979886449697775903164251909656053550902322568838529801236463172863864, 154);
        node_index.insert(96230476333472367292088003479392846911668311618815845294471665603593864083, 155);
        node_index.insert(1369434440767862319960208049698504788320960774232879724280117316982534650193, 156);
        node_index.insert(2024621847884265898218489264737458834270061187194941746620500787079400176884, 157);
        node_index.insert(1807505530087998551854838238942081975706714981342010316493930295188666464940, 158);
        node_index.insert(1256426694963459690523405474782682900655353255288895132116069789184783004863, 159);
        node_index.insert(1480942784604782481947935934901384162247989888269331261880373449270545013773, 160);
        node_index.insert(1517285499547985653662412497830230693164818630246912917362097237933215286769, 161);
        node_index.insert(1775850300486095625315352560841918103386799050783892484180849210397312047913, 162);
        node_index.insert(2128745048370263957300510818330711821104917422838702078718700125278796610857, 163);
        node_index.insert(2052253867188246865634385442842098208489880204748745318795500528396002438643, 164);
        node_index.insert(823811640053860064462641613549091334297367639108581677320681253170766237119, 165);
        node_index.insert(3179889427390602460685268655663400734144277245525156921438940049219220298472, 166);
        node_index.insert(1419937426584623687818364410196464090212535066425540720870185944025022347854, 167);
        node_index.insert(2192635961575105440357479419879570115317354650047551283600164572889850391397, 168);
        node_index.insert(3010441263787617485144232164262690884517292459461887802066090347178160489446, 169);
        node_index.insert(1575202856616858204066291142973659090691398987707763324629342621278044048285, 170);
        node_index.insert(3150767609235658512538174793063598453685235159814426872421706497892360993473, 171);
        node_index.insert(1380581146511630386393189541396220147704823992328592397504084198853794580698, 172);
        node_index.insert(3546631549955366734370849149420831006324559095944560863653820622232224546550, 173);
        node_index.insert(2810431988481695461417148219596043474790991687777393392127083786467414097023, 174);
        node_index.insert(2668303977292109594218165549431706660277475416304558283917935571063120119619, 175);
        node_index.insert(775077831086268747685749623039943718352069980301731753822172089211081968004, 176);
        node_index.insert(3090788003186441392649192821414504889094443187145074321087442086807171723091, 177);
        node_index.insert(1747877719494878675330654512691378207436164258900560542576645030101944654198, 178);
        node_index.insert(1855807191027959492272087099093127895996925391310249384853724014589354065880, 179);
        node_index.insert(1129931679721407382546843666485768437924105660568409460122129529320833750538, 180);
        node_index.insert(615190837154554050699251218174262050387027596382481883911392248587222883039, 181);
        node_index.insert(2734385063554876165865685967765135790485015557637609746899130055333350058131, 182);
        node_index.insert(3300801161201855210543983895929417281565856301714486697683966619425717591788, 183);
        node_index.insert(606689980960819825217599323540711634241712783276063345307862820473909229433, 184);
        node_index.insert(1435517226781720725534280962421721726680639579406355452425224073262988429300, 185);
        node_index.insert(218090268678609515304322263152602616587525765268609577600069603942431353921, 186);
        node_index.insert(1482455204544669688530906145322525078153575308477782681500284850271921656123, 187);
        node_index.insert(2723538244488124685018228519562656844025444137189174725592533236091714933662, 188);
        node_index.insert(1416765255842618003923310557773125835035600865242012452755344332853237071588, 189);
        node_index.insert(2160818454084740153200984358082017003049189165070506564327863557823286418982, 190);
        node_index.insert(2277613560213189153618961724214214762768786100426878373029659162621843745947, 191);
        node_index.insert(2310985831016527007003204927936454453271291287169255250680051621970559607259, 192);
        node_index.insert(384116465256348490432262897955508326078213593152164252876243767071765438369, 193);
        node_index.insert(1702928269486420556114444750664242098751524332365203170460523987041089838264, 194);
        node_index.insert(1737969280824721161290349436642041003594826887387004385941684426614913572142, 195);
        node_index.insert(1836781227583069779219160796082107414389044990588064523362453501233348447941, 196);
        node_index.insert(1835413265224915965797476997991004024975537945172393162125625790064443576056, 197);
        node_index.insert(3577969631049740847270927026689997982663987892468748730512805116117892393850, 198);
        node_index.insert(125184167455727684497312111598873828527136579269573677836666274982798717502, 199);
        node_index.insert(916920587948602885142894279978223961143345580053232718370266827220007051042, 200);
        node_index.insert(1135558104669773446199788744536203987648820209155634743614295303291149076972, 201);
        node_index.insert(569319179470172407891644686289133669143857652922477416322603752001435538727, 202);
        node_index.insert(28144270457860219994540784271960633810341592210783112627495867462125341399, 203);
        node_index.insert(18799720310179697131131685103073374725225088093137210429877994125787778970, 204);
        node_index.insert(1907592993586083015232420056709823715260089051436724566279976031737880496896, 205);
        node_index.insert(1325857619669285999672217577689034662554765538380980402699076229699716710354, 206);
        node_index.insert(3534592123424374672525832747636041212144498252558026446919832375257794754078, 207);
        node_index.insert(975811662252404486364210566960617261119194048048057753124467019139043082780, 208);
        node_index.insert(1203783440064645674993999134182043448681625959197437382492961638345396223736, 209);
        node_index.insert(2965915082841270961849114004997788133178882537752364543376213115956218668068, 210);
        node_index.insert(2196678506148716094359840762843095294547431304170281593313213840356758681995, 211);
        node_index.insert(2005813385367076411036344259477321710390165026780598801785268103266785260257, 212);
        node_index.insert(812670197976491819516894685617320108477880322210855848068169089021149996096, 213);
        node_index.insert(3027346615299202831427725205593487015338134891720614685771024785557600089179, 214);
        node_index.insert(2262141666650713711957596690663865336482476990485123650378827031749065539153, 215);
        node_index.insert(1400540991080624038172644934842794351966643174543989040900762912239368862046, 216);
        node_index.insert(2674007638051766565985483780338803878333369059554977104285252505360641540495, 217);
        node_index.insert(421562970220145069512752674992305658674527798369971776682952067577148320293, 218);
        node_index.insert(2538381879635566632080772386938834217090950791368259623569067719971406576562, 219);
        node_index.insert(785250261230865465845710679769065027126620330511677439151523279843384735707, 220);
        node_index.insert(2675774331086498540624933902003307031193934996193179623025584398058530854168, 221);
        node_index.insert(2029231873060600943780848288222784448626407095295744170402030958579757213210, 222);
        node_index.insert(1369682932505030338857908219046067199211450301530342842349893155867238618958, 223);
        node_index.insert(1617787748514986090736380478953387380692432482267497991595252403068144096248, 224);
        node_index.insert(298581901040189072278001148796869436800005707712750464200573558400196355794, 225);
        node_index.insert(2915780834624707756184844711775801600271458380800621795631283771448087489188, 226);
        node_index.insert(1726726174168495543694837532694031638106884725300692897396886304872712092148, 227);
        node_index.insert(564104024341316111401760557020882561501025314369970835483370897970254325207, 228);
        node_index.insert(3080725628534418360139430407210520788655436992406730489445684097308935812337, 229);
        node_index.insert(3490933986396213215220335172083634022587701690218130990803729683092419978220, 230);
        node_index.insert(3341687902555355867089532501146030994312759720542274425366422090162764510428, 231);
        node_index.insert(1199411263819779954197012371115092851316169638939517069714036706528168468239, 232);
        node_index.insert(27989182845787072097190432684511558532229547537947259013324298054968001724, 233);

    let atts = TreeEnsembleAttributes {
        nodes_falsenodeids,
        nodes_featureids,
        nodes_missing_value_tracks_true,
        nodes_modes,
        nodes_nodeids,
        nodes_treeids,
        nodes_truenodeids,
        nodes_values
    };

    let mut ensemble: TreeEnsemble<FP16x16> = TreeEnsemble {
        atts, tree_ids, root_index, node_index
    };

    let mut classifier: TreeEnsembleClassifier<FP16x16> = TreeEnsembleClassifier {
        ensemble,
        class_ids,
        class_nodeids,
        class_treeids,
        class_weights,
        classlabels,
        base_values,
        post_transform
    };

    let (top_class, mut scores) = ml::TreeEnsembleClassifierTrait::predict(classifier, node_input);
    let mut data: Array<FP16x16> = array![];
    let mut i: usize = 0;
    let data_len = scores.data.len;
    while i < data_len {
        data.append(scores.data.at(i));

        i += 1;
    };
    println!("{}", @scores.get(0, 0).unwrap().mag);
    println!("{}", @scores.get(0, 1).unwrap().mag);
    println!("{}", @scores.get(0, 2).unwrap().mag);
    println!("{}", @scores.get(0, 3).unwrap().mag);
    (top_class ,Tensor {shape: array![scores.rows, scores.cols].span(), data: data.span()})

}