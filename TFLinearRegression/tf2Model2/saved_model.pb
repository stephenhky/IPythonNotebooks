ͱ
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�"serve*2.0.02v2.0.0-rc2-26-g64c3d382ca8��
^
WVarHandleOp*
shared_nameW*
dtype0*
_output_shapes
: *
shape
:
W
W/Read/ReadVariableOpReadVariableOpW*
dtype0*
_output_shapes

:
Z
bVarHandleOp*
shape:*
shared_nameb*
dtype0*
_output_shapes
: 
S
b/Read/ReadVariableOpReadVariableOpb*
dtype0*
_output_shapes
:
^
totalVarHandleOp*
shared_nametotal*
dtype0*
_output_shapes
: *
shape: 
W
total/Read/ReadVariableOpReadVariableOptotal*
dtype0*
_output_shapes
: 
^
countVarHandleOp*
dtype0*
_output_shapes
: *
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
dtype0*
_output_shapes
: 
h

Nadam/iterVarHandleOp*
shared_name
Nadam/iter*
dtype0	*
_output_shapes
: *
shape: 
a
Nadam/iter/Read/ReadVariableOpReadVariableOp
Nadam/iter*
dtype0	*
_output_shapes
: 
l
Nadam/beta_1VarHandleOp*
dtype0*
_output_shapes
: *
shape: *
shared_nameNadam/beta_1
e
 Nadam/beta_1/Read/ReadVariableOpReadVariableOpNadam/beta_1*
dtype0*
_output_shapes
: 
l
Nadam/beta_2VarHandleOp*
shape: *
shared_nameNadam/beta_2*
dtype0*
_output_shapes
: 
e
 Nadam/beta_2/Read/ReadVariableOpReadVariableOpNadam/beta_2*
dtype0*
_output_shapes
: 
j
Nadam/decayVarHandleOp*
shared_nameNadam/decay*
dtype0*
_output_shapes
: *
shape: 
c
Nadam/decay/Read/ReadVariableOpReadVariableOpNadam/decay*
dtype0*
_output_shapes
: 
z
Nadam/learning_rateVarHandleOp*$
shared_nameNadam/learning_rate*
dtype0*
_output_shapes
: *
shape: 
s
'Nadam/learning_rate/Read/ReadVariableOpReadVariableOpNadam/learning_rate*
dtype0*
_output_shapes
: 
|
Nadam/momentum_cacheVarHandleOp*
shape: *%
shared_nameNadam/momentum_cache*
dtype0*
_output_shapes
: 
u
(Nadam/momentum_cache/Read/ReadVariableOpReadVariableOpNadam/momentum_cache*
dtype0*
_output_shapes
: 
n
	Nadam/W/mVarHandleOp*
dtype0*
_output_shapes
: *
shape
:*
shared_name	Nadam/W/m
g
Nadam/W/m/Read/ReadVariableOpReadVariableOp	Nadam/W/m*
dtype0*
_output_shapes

:
j
	Nadam/b/mVarHandleOp*
shared_name	Nadam/b/m*
dtype0*
_output_shapes
: *
shape:
c
Nadam/b/m/Read/ReadVariableOpReadVariableOp	Nadam/b/m*
dtype0*
_output_shapes
:
n
	Nadam/W/vVarHandleOp*
dtype0*
_output_shapes
: *
shape
:*
shared_name	Nadam/W/v
g
Nadam/W/v/Read/ReadVariableOpReadVariableOp	Nadam/W/v*
dtype0*
_output_shapes

:
j
	Nadam/b/vVarHandleOp*
shape:*
shared_name	Nadam/b/v*
dtype0*
_output_shapes
: 
c
Nadam/b/v/Read/ReadVariableOpReadVariableOp	Nadam/b/v*
dtype0*
_output_shapes
:

NoOpNoOp
�
ConstConst"/device:CPU:0*�
value�B� B�
9
W
b

lrLoss
	optimizer

signatures
31
VARIABLE_VALUEWW/.ATTRIBUTES/VARIABLE_VALUE
31
VARIABLE_VALUEbb/.ATTRIBUTES/VARIABLE_VALUE
x
	total
	count

_fn_kwargs
	regularization_losses

trainable_variables
	variables
	keras_api
x
iter

beta_1

beta_2
	decay
learning_rate
momentum_cachemmvv
 
B@
VARIABLE_VALUEtotal'lrLoss/total/.ATTRIBUTES/VARIABLE_VALUE
B@
VARIABLE_VALUEcount'lrLoss/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

0
1
�
non_trainable_variables
	regularization_losses
layer_regularization_losses

trainable_variables
	variables
metrics

layers
IG
VARIABLE_VALUE
Nadam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUENadam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUENadam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUENadam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUENadam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUENadam/momentum_cache3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUE

0
1
 
 
 
WU
VARIABLE_VALUE	Nadam/W/m8W/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUE	Nadam/b/m8b/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUE	Nadam/W/v8W/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUE	Nadam/b/v8b/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
: 
O
saver_filenamePlaceholder*
dtype0*
_output_shapes
: *
shape: 
�
StatefulPartitionedCallStatefulPartitionedCallsaver_filenameW/Read/ReadVariableOpb/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpNadam/iter/Read/ReadVariableOp Nadam/beta_1/Read/ReadVariableOp Nadam/beta_2/Read/ReadVariableOpNadam/decay/Read/ReadVariableOp'Nadam/learning_rate/Read/ReadVariableOp(Nadam/momentum_cache/Read/ReadVariableOpNadam/W/m/Read/ReadVariableOpNadam/b/m/Read/ReadVariableOpNadam/W/v/Read/ReadVariableOpNadam/b/v/Read/ReadVariableOpConst*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2	*
_output_shapes
: *+
_gradient_op_typePartitionedCall-1759*&
f!R
__inference__traced_save_1758
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameWbtotalcount
Nadam/iterNadam/beta_1Nadam/beta_2Nadam/decayNadam/learning_rateNadam/momentum_cache	Nadam/W/m	Nadam/b/m	Nadam/W/v	Nadam/b/v*
Tout
2**
config_proto

CPU

GPU 2J 8*
_output_shapes
: *
Tin
2*+
_gradient_op_typePartitionedCall-1814*)
f$R"
 __inference__traced_restore_1813��
�7
�
 __inference__traced_restore_1813
file_prefix
assignvariableop_w
assignvariableop_1_b
assignvariableop_2_total
assignvariableop_3_count!
assignvariableop_4_nadam_iter#
assignvariableop_5_nadam_beta_1#
assignvariableop_6_nadam_beta_2"
assignvariableop_7_nadam_decay*
&assignvariableop_8_nadam_learning_rate+
'assignvariableop_9_nadam_momentum_cache!
assignvariableop_10_nadam_w_m!
assignvariableop_11_nadam_b_m!
assignvariableop_12_nadam_w_v!
assignvariableop_13_nadam_b_v
identity_15��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�	RestoreV2�RestoreV2_1�
RestoreV2/tensor_namesConst"/device:CPU:0*�
value�B�BW/.ATTRIBUTES/VARIABLE_VALUEBb/.ATTRIBUTES/VARIABLE_VALUEB'lrLoss/total/.ATTRIBUTES/VARIABLE_VALUEB'lrLoss/count/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB8W/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB8b/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB8W/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB8b/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*/
value&B$B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
dtypes
2	*L
_output_shapes:
8::::::::::::::L
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:n
AssignVariableOpAssignVariableOpassignvariableop_wIdentity:output:0*
dtype0*
_output_shapes
 N

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:t
AssignVariableOp_1AssignVariableOpassignvariableop_1_bIdentity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
_output_shapes
:*
T0x
AssignVariableOp_2AssignVariableOpassignvariableop_2_totalIdentity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:x
AssignVariableOp_3AssignVariableOpassignvariableop_3_countIdentity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
T0	*
_output_shapes
:}
AssignVariableOp_4AssignVariableOpassignvariableop_4_nadam_iterIdentity_4:output:0*
dtype0	*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_nadam_beta_1Identity_5:output:0*
dtype0*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOpassignvariableop_6_nadam_beta_2Identity_6:output:0*
_output_shapes
 *
dtype0N

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:~
AssignVariableOp_7AssignVariableOpassignvariableop_7_nadam_decayIdentity_7:output:0*
dtype0*
_output_shapes
 N

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp&assignvariableop_8_nadam_learning_rateIdentity_8:output:0*
dtype0*
_output_shapes
 N

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp'assignvariableop_9_nadam_momentum_cacheIdentity_9:output:0*
_output_shapes
 *
dtype0P
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_nadam_w_mIdentity_10:output:0*
dtype0*
_output_shapes
 P
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_nadam_b_mIdentity_11:output:0*
dtype0*
_output_shapes
 P
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_nadam_w_vIdentity_12:output:0*
_output_shapes
 *
dtype0P
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_nadam_b_vIdentity_13:output:0*
_output_shapes
 *
dtype0�
RestoreV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:�
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
dtypes
2*
_output_shapes
:1
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_14Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: �
Identity_15IdentityIdentity_14:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
_output_shapes
: *
T0"#
identity_15Identity_15:output:0*M
_input_shapes<
:: ::::::::::::::2(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122
RestoreV2_1RestoreV2_12*
AssignVariableOp_13AssignVariableOp_132(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_2: : : :	 :
 : : : : :+ '
%
_user_specified_namefile_prefix: : : : : 
��
�
__inference_train_step_1694
vectors

labels"
matmul_readvariableop_resource
add_readvariableop_resource*
&nadam_identity_readvariableop_resource&
"nadam_cast_readvariableop_resource(
$nadam_cast_1_readvariableop_resource(
$nadam_cast_2_readvariableop_resource!
nadam_readvariableop_resource2
.nadam_nadam_update_mul_readvariableop_resource4
0nadam_nadam_update_mul_2_readvariableop_resource4
0nadam_nadam_update_1_mul_readvariableop_resource6
2nadam_nadam_update_1_mul_2_readvariableop_resource��MatMul/ReadVariableOp�Nadam/AssignVariableOp�Nadam/Cast/ReadVariableOp�Nadam/Cast_1/ReadVariableOp�Nadam/Cast_2/ReadVariableOp�Nadam/Identity/ReadVariableOp�Nadam/Identity_4/ReadVariableOp�Nadam/Nadam/AssignAddVariableOp�#Nadam/Nadam/update/AssignVariableOp�%Nadam/Nadam/update/AssignVariableOp_1�%Nadam/Nadam/update/AssignVariableOp_2�&Nadam/Nadam/update/Read/ReadVariableOp�!Nadam/Nadam/update/ReadVariableOp�#Nadam/Nadam/update/ReadVariableOp_1�#Nadam/Nadam/update/ReadVariableOp_2�%Nadam/Nadam/update/mul/ReadVariableOp�'Nadam/Nadam/update/mul_2/ReadVariableOp�%Nadam/Nadam/update_1/AssignVariableOp�'Nadam/Nadam/update_1/AssignVariableOp_1�'Nadam/Nadam/update_1/AssignVariableOp_2�(Nadam/Nadam/update_1/Read/ReadVariableOp�#Nadam/Nadam/update_1/ReadVariableOp�%Nadam/Nadam/update_1/ReadVariableOp_1�%Nadam/Nadam/update_1/ReadVariableOp_2�'Nadam/Nadam/update_1/mul/ReadVariableOp�)Nadam/Nadam/update_1/mul_2/ReadVariableOp�Nadam/ReadVariableOp�Nadam/ReadVariableOp_1�add/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:b
MatMulMatMulvectorsMatMul/ReadVariableOp:value:0*
_output_shapes
:	�N*
T0�
add/ReadVariableOpReadVariableOpadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:d
addAddV2MatMul:product:0add/ReadVariableOp:value:0*
T0*
_output_shapes
:	�Nt
$mean_squared_error/SquaredDifferenceSquaredDifferenceadd:z:0labels*
_output_shapes
:	�N*
T0t
)mean_squared_error/Mean/reduction_indicesConst*
valueB :
���������*
dtype0*
_output_shapes
: �
mean_squared_error/MeanMean(mean_squared_error/SquaredDifference:z:02mean_squared_error/Mean/reduction_indices:output:0*
T0*
_output_shapes	
:�Nl
'mean_squared_error/weighted_loss/Cast/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: �
Umean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/weights/shapeConst*
_output_shapes
: *
valueB *
dtype0�
Tmean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: �
Tmean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/values/shapeConst*
valueB:�N*
dtype0*
_output_shapes
:�
Smean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/values/rankConst*
dtype0*
_output_shapes
: *
value	B :�
cmean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOp*
_output_shapes
 �
Bmean_squared_error/weighted_loss/broadcast_weights/ones_like/ShapeConstd^mean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB:�N*
dtype0*
_output_shapes
:�
Bmean_squared_error/weighted_loss/broadcast_weights/ones_like/ConstConstd^mean_squared_error/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  �?*
dtype0*
_output_shapes
: �
<mean_squared_error/weighted_loss/broadcast_weights/ones_likeFillKmean_squared_error/weighted_loss/broadcast_weights/ones_like/Shape:output:0Kmean_squared_error/weighted_loss/broadcast_weights/ones_like/Const:output:0*
_output_shapes	
:�N*
T0�
2mean_squared_error/weighted_loss/broadcast_weightsMul0mean_squared_error/weighted_loss/Cast/x:output:0Emean_squared_error/weighted_loss/broadcast_weights/ones_like:output:0*
T0*
_output_shapes	
:�N�
$mean_squared_error/weighted_loss/MulMul mean_squared_error/Mean:output:06mean_squared_error/weighted_loss/broadcast_weights:z:0*
T0*
_output_shapes	
:�Np
&mean_squared_error/weighted_loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:�
$mean_squared_error/weighted_loss/SumSum(mean_squared_error/weighted_loss/Mul:z:0/mean_squared_error/weighted_loss/Const:output:0*
T0*
_output_shapes
: p
-mean_squared_error/weighted_loss/num_elementsConst*
value
B :�N*
dtype0*
_output_shapes
: �
2mean_squared_error/weighted_loss/num_elements/CastCast6mean_squared_error/weighted_loss/num_elements:output:0*

SrcT0*

DstT0*
_output_shapes
: k
(mean_squared_error/weighted_loss/Const_1Const*
valueB *
dtype0*
_output_shapes
: �
&mean_squared_error/weighted_loss/Sum_1Sum-mean_squared_error/weighted_loss/Sum:output:01mean_squared_error/weighted_loss/Const_1:output:0*
_output_shapes
: *
T0�
&mean_squared_error/weighted_loss/valueDivNoNan/mean_squared_error/weighted_loss/Sum_1:output:06mean_squared_error/weighted_loss/num_elements/Cast:y:0*
_output_shapes
: *
T0I
onesConst*
dtype0*
_output_shapes
: *
valueB
 *  �?H
ShapeConst*
valueB *
dtype0*
_output_shapes
: J
Shape_1Const*
valueB *
dtype0*
_output_shapes
: �
BroadcastGradientArgsBroadcastGradientArgsShape:output:0Shape_1:output:0*2
_output_shapes 
:���������:���������~

div_no_nanDivNoNanones:output:06mean_squared_error/weighted_loss/num_elements/Cast:y:0*
T0*
_output_shapes
: W
SumSumdiv_no_nan:z:0BroadcastGradientArgs:r0:0*
_output_shapes
: *
T0Q
ReshapeReshapeSum:output:0Shape:output:0*
T0*
_output_shapes
: \
NegNeg/mean_squared_error/weighted_loss/Sum_1:output:0*
T0*
_output_shapes
: z
div_no_nan_1DivNoNanNeg:y:06mean_squared_error/weighted_loss/num_elements/Cast:y:0*
T0*
_output_shapes
: �
div_no_nan_2DivNoNandiv_no_nan_1:z:06mean_squared_error/weighted_loss/num_elements/Cast:y:0*
_output_shapes
: *
T0L
mulMulones:output:0div_no_nan_2:z:0*
T0*
_output_shapes
: R
Sum_1Summul:z:0BroadcastGradientArgs:r1:0*
T0*
_output_shapes
: W
	Reshape_1ReshapeSum_1:output:0Shape_1:output:0*
T0*
_output_shapes
: R
Reshape_2/shapeConst*
valueB *
dtype0*
_output_shapes
: a
	Reshape_2ReshapeReshape:output:0Reshape_2/shape:output:0*
T0*
_output_shapes
: H
ConstConst*
valueB *
dtype0*
_output_shapes
: Q
TileTileReshape_2:output:0Const:output:0*
T0*
_output_shapes
: Y
Reshape_3/shapeConst*
valueB:*
dtype0*
_output_shapes
:b
	Reshape_3ReshapeTile:output:0Reshape_3/shape:output:0*
_output_shapes
:*
T0R
Const_1Const*
valueB:�N*
dtype0*
_output_shapes
:Z
Tile_1TileReshape_3:output:0Const_1:output:0*
T0*
_output_shapes	
:�N{
Mul_1MulTile_1:output:06mean_squared_error/weighted_loss/broadcast_weights:z:0*
T0*
_output_shapes	
:�Ne
Mul_2MulTile_1:output:0 mean_squared_error/Mean:output:0*
_output_shapes	
:�N*
T0W
Cast/xConst*
valueB"'     *
dtype0*
_output_shapes
:[
Cast_1/xConst*
valueB:
���������*
dtype0*
_output_shapes
:F
SizeConst*
_output_shapes
: *
value	B :*
dtype0U
add_1AddV2Cast_1/x:output:0Size:output:0*
T0*
_output_shapes
:N
modFloorMod	add_1:z:0Size:output:0*
T0*
_output_shapes
:Q
Shape_2Const*
valueB:*
dtype0*
_output_shapes
:M
range/startConst*
dtype0*
_output_shapes
: *
value	B : M
range/deltaConst*
value	B :*
dtype0*
_output_shapes
: e
rangeRangerange/start:output:0Size:output:0range/delta:output:0*
_output_shapes
:L

Fill/valueConst*
_output_shapes
: *
value	B :*
dtype0X
FillFillShape_2:output:0Fill/value:output:0*
_output_shapes
:*
T0�
DynamicStitchDynamicStitchrange:output:0mod:z:0Cast/x:output:0Fill:output:0*
N*
_output_shapes
:*
T0Z
	Maximum/xConst*
valueB"'     *
dtype0*
_output_shapes
:K
	Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: _
MaximumMaximumMaximum/x:output:0Maximum/y:output:0*
T0*
_output_shapes
:[

floordiv/xConst*
dtype0*
_output_shapes
:*
valueB"'     [
floordivFloorDivfloordiv/x:output:0Maximum:z:0*
_output_shapes
:*
T0`
Reshape_4/shapeConst*
valueB"'     *
dtype0*
_output_shapes
:c
	Reshape_4Reshape	Mul_1:z:0Reshape_4/shape:output:0*
T0*
_output_shapes
:	�Na
Tile_2/multiplesConst*
valueB"      *
dtype0*
_output_shapes
:g
Tile_2TileReshape_4:output:0Tile_2/multiples:output:0*
T0*
_output_shapes
:	�NL
Const_2Const*
dtype0*
_output_shapes
: *
valueB
 *  �?_
truedivRealDivTile_2:output:0Const_2:output:0*
_output_shapes
:	�N*
T0U
scalarConst^truediv*
valueB
 *   @*
dtype0*
_output_shapes
: T
Mul_3Mulscalar:output:0truediv:z:0*
T0*
_output_shapes
:	�NO
subSubadd:z:0labels^truediv*
T0*
_output_shapes
:	�NJ
mul_4Mul	Mul_3:z:0sub:z:0*
_output_shapes
:	�N*
T0A
Neg_1Neg	mul_4:z:0*
T0*
_output_shapes
:	�Nk
BroadcastGradientArgs_1/s0Const*
valueB"'     *
dtype0*
_output_shapes
:d
BroadcastGradientArgs_1/s1Const*
dtype0*
_output_shapes
:*
valueB:�
BroadcastGradientArgs_1BroadcastGradientArgs#BroadcastGradientArgs_1/s0:output:0#BroadcastGradientArgs_1/s1:output:0*2
_output_shapes 
:���������:���������h
Sum_2/reduction_indicesConst*
valueB"       *
dtype0*
_output_shapes
:Z
Sum_2Sum	mul_4:z:0 Sum_2/reduction_indices:output:0*
_output_shapes
: *
T0Y
Reshape_5/shapeConst*
valueB:*
dtype0*
_output_shapes
:c
	Reshape_5ReshapeSum_2:output:0Reshape_5/shape:output:0*
T0*
_output_shapes
:b
MatMul_1MatMulvectors	mul_4:z:0*
T0*
transpose_a(*
_output_shapes

:�
Nadam/Identity/ReadVariableOpReadVariableOp&nadam_identity_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
: b
Nadam/IdentityIdentity%Nadam/Identity/ReadVariableOp:value:0*
_output_shapes
: *
T0�
Nadam/Cast/ReadVariableOpReadVariableOp"nadam_cast_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
: �
Nadam/Identity_1Identity!Nadam/Cast/ReadVariableOp:value:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
T0�
Nadam/Cast_1/ReadVariableOpReadVariableOp$nadam_cast_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
dtype0�
Nadam/Identity_2Identity#Nadam/Cast_1/ReadVariableOp:value:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
T0�
Nadam/Cast_2/ReadVariableOpReadVariableOp$nadam_cast_2_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
: �
Nadam/Identity_3Identity#Nadam/Cast_2/ReadVariableOp:value:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
T0�
Nadam/ReadVariableOpReadVariableOpnadam_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0	*
_output_shapes
: {
Nadam/add/yConst",/job:localhost/replica:0/task:0/device:CPU:0*
value	B	 R*
dtype0	*
_output_shapes
: �
	Nadam/addAddV2Nadam/ReadVariableOp:value:0Nadam/add/y:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0	*
_output_shapes
: �
Nadam/Cast_3CastNadam/add:z:0",/job:localhost/replica:0/task:0/device:CPU:0*

SrcT0	*

DstT0*
_output_shapes
: �
Nadam/ReadVariableOp_1ReadVariableOpnadam_readvariableop_resource^Nadam/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0	*
_output_shapes
: }
Nadam/add_1/yConst",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0	*
_output_shapes
: *
value	B	 R�
Nadam/add_1AddV2Nadam/ReadVariableOp_1:value:0Nadam/add_1/y:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0	*
_output_shapes
: �
Nadam/Cast_4CastNadam/add_1:z:0",/job:localhost/replica:0/task:0/device:CPU:0*

DstT0*
_output_shapes
: *

SrcT0	�
Nadam/Cast_5/xConst",/job:localhost/replica:0/task:0/device:CPU:0*
valueB
 *��u?*
dtype0*
_output_shapes
: ~
Nadam/mul/xConst",/job:localhost/replica:0/task:0/device:CPU:0*
valueB
 *o�;*
dtype0*
_output_shapes
: �
	Nadam/mulMulNadam/mul/x:output:0Nadam/Cast_3:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
T0�
	Nadam/PowPowNadam/Cast_5/x:output:0Nadam/mul:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes
: �
Nadam/mul_1/xConst",/job:localhost/replica:0/task:0/device:CPU:0*
valueB
 *   ?*
dtype0*
_output_shapes
: �
Nadam/mul_1MulNadam/mul_1/x:output:0Nadam/Pow:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
T0~
Nadam/sub/xConst",/job:localhost/replica:0/task:0/device:CPU:0*
valueB
 *  �?*
dtype0*
_output_shapes
: �
	Nadam/subSubNadam/sub/x:output:0Nadam/mul_1:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
T0�
Nadam/mul_2MulNadam/Identity_2:output:0Nadam/sub:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes
: �
Nadam/mul_3/xConst",/job:localhost/replica:0/task:0/device:CPU:0*
valueB
 *o�;*
dtype0*
_output_shapes
: �
Nadam/mul_3MulNadam/mul_3/x:output:0Nadam/Cast_4:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
T0�
Nadam/Pow_1PowNadam/Cast_5/x:output:0Nadam/mul_3:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
T0�
Nadam/mul_4/xConst",/job:localhost/replica:0/task:0/device:CPU:0*
valueB
 *   ?*
dtype0*
_output_shapes
: �
Nadam/mul_4MulNadam/mul_4/x:output:0Nadam/Pow_1:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes
: �
Nadam/sub_1/xConst",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
valueB
 *  �?*
dtype0�
Nadam/sub_1SubNadam/sub_1/x:output:0Nadam/mul_4:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes
: �
Nadam/mul_5MulNadam/Identity_2:output:0Nadam/sub_1:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes
: �
Nadam/mul_6MulNadam/Identity:output:0Nadam/mul_2:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes
: �
Nadam/AssignVariableOpAssignVariableOp&nadam_identity_readvariableop_resourceNadam/mul_6:z:0^Nadam/Identity/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
 �
Nadam/Identity_4/ReadVariableOpReadVariableOp&nadam_identity_readvariableop_resource^Nadam/AssignVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
: �
Nadam/Identity_4Identity'Nadam/Identity_4/ReadVariableOp:value:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes
: �
Nadam/mul_7MulNadam/Identity_4:output:0Nadam/mul_5:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes
: z
	Nadam/NegNegNadam/Identity_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes
: ~
Nadam/ConstConst",/job:localhost/replica:0/task:0/device:CPU:0*
valueB
 *���3*
dtype0*
_output_shapes
: �
Nadam/sub_2/xConst",/job:localhost/replica:0/task:0/device:CPU:0*
valueB
 *  �?*
dtype0*
_output_shapes
: �
Nadam/sub_2SubNadam/sub_2/x:output:0Nadam/Identity_2:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes
: �
Nadam/sub_3/xConst",/job:localhost/replica:0/task:0/device:CPU:0*
valueB
 *  �?*
dtype0*
_output_shapes
: �
Nadam/sub_3SubNadam/sub_3/x:output:0Nadam/Identity_3:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
T0�
Nadam/sub_4/xConst",/job:localhost/replica:0/task:0/device:CPU:0*
valueB
 *  �?*
dtype0*
_output_shapes
: �
Nadam/sub_4SubNadam/sub_4/x:output:0Nadam/mul_2:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes
: �
Nadam/sub_5/xConst",/job:localhost/replica:0/task:0/device:CPU:0*
valueB
 *  �?*
dtype0*
_output_shapes
: �
Nadam/sub_5SubNadam/sub_5/x:output:0Nadam/Identity_4:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
T0�
Nadam/sub_6/xConst",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
valueB
 *  �?*
dtype0�
Nadam/sub_6SubNadam/sub_6/x:output:0Nadam/mul_7:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
T0�
Nadam/Pow_2PowNadam/Identity_3:output:0Nadam/Cast_3:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
T0�
Nadam/sub_7/xConst",/job:localhost/replica:0/task:0/device:CPU:0*
valueB
 *  �?*
dtype0*
_output_shapes
: �
Nadam/sub_7SubNadam/sub_7/x:output:0Nadam/Pow_2:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes
: �
&Nadam/Nadam/update/Read/ReadVariableOpReadVariableOpmatmul_readvariableop_resource^MatMul/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:�
Nadam/Nadam/update/IdentityIdentity.Nadam/Nadam/update/Read/ReadVariableOp:value:0*
T0*
_output_shapes

:�
Nadam/Nadam/update/truedivRealDivMatMul_1:product:0Nadam/sub_5:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*9
_class/
-+loc:@Nadam/Nadam/update/Read/ReadVariableOp*
_output_shapes

:�
%Nadam/Nadam/update/mul/ReadVariableOpReadVariableOp.nadam_nadam_update_mul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:�
Nadam/Nadam/update/mulMulNadam/Identity_2:output:0-Nadam/Nadam/update/mul/ReadVariableOp:value:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:*
T0*9
_class/
-+loc:@Nadam/Nadam/update/Read/ReadVariableOp�
Nadam/Nadam/update/mul_1MulNadam/sub_2:z:0MatMul_1:product:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*9
_class/
-+loc:@Nadam/Nadam/update/Read/ReadVariableOp*
_output_shapes

:�
Nadam/Nadam/update/addAddV2Nadam/Nadam/update/mul:z:0Nadam/Nadam/update/mul_1:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*9
_class/
-+loc:@Nadam/Nadam/update/Read/ReadVariableOp*
_output_shapes

:�
#Nadam/Nadam/update/AssignVariableOpAssignVariableOp.nadam_nadam_update_mul_readvariableop_resourceNadam/Nadam/update/add:z:0&^Nadam/Nadam/update/mul/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*9
_class/
-+loc:@Nadam/Nadam/update/Read/ReadVariableOp*
dtype0*
_output_shapes
 �
!Nadam/Nadam/update/ReadVariableOpReadVariableOp.nadam_nadam_update_mul_readvariableop_resource$^Nadam/Nadam/update/AssignVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*9
_class/
-+loc:@Nadam/Nadam/update/Read/ReadVariableOp*
dtype0*
_output_shapes

:�
Nadam/Nadam/update/truediv_1RealDiv)Nadam/Nadam/update/ReadVariableOp:value:0Nadam/sub_6:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*9
_class/
-+loc:@Nadam/Nadam/update/Read/ReadVariableOp*
_output_shapes

:�
'Nadam/Nadam/update/mul_2/ReadVariableOpReadVariableOp0nadam_nadam_update_mul_2_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:�
Nadam/Nadam/update/mul_2MulNadam/Identity_3:output:0/Nadam/Nadam/update/mul_2/ReadVariableOp:value:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*9
_class/
-+loc:@Nadam/Nadam/update/Read/ReadVariableOp*
_output_shapes

:�
Nadam/Nadam/update/SquareSquareMatMul_1:product:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*9
_class/
-+loc:@Nadam/Nadam/update/Read/ReadVariableOp*
_output_shapes

:�
Nadam/Nadam/update/mul_3MulNadam/sub_3:z:0Nadam/Nadam/update/Square:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*9
_class/
-+loc:@Nadam/Nadam/update/Read/ReadVariableOp*
_output_shapes

:�
Nadam/Nadam/update/add_1AddV2Nadam/Nadam/update/mul_2:z:0Nadam/Nadam/update/mul_3:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:*
T0*9
_class/
-+loc:@Nadam/Nadam/update/Read/ReadVariableOp�
%Nadam/Nadam/update/AssignVariableOp_1AssignVariableOp0nadam_nadam_update_mul_2_readvariableop_resourceNadam/Nadam/update/add_1:z:0(^Nadam/Nadam/update/mul_2/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*9
_class/
-+loc:@Nadam/Nadam/update/Read/ReadVariableOp*
dtype0*
_output_shapes
 �
#Nadam/Nadam/update/ReadVariableOp_1ReadVariableOp0nadam_nadam_update_mul_2_readvariableop_resource&^Nadam/Nadam/update/AssignVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*9
_class/
-+loc:@Nadam/Nadam/update/Read/ReadVariableOp*
dtype0*
_output_shapes

:�
Nadam/Nadam/update/truediv_2RealDiv+Nadam/Nadam/update/ReadVariableOp_1:value:0Nadam/sub_7:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*9
_class/
-+loc:@Nadam/Nadam/update/Read/ReadVariableOp*
_output_shapes

:�
Nadam/Nadam/update/mul_4MulNadam/sub_4:z:0Nadam/Nadam/update/truediv:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*9
_class/
-+loc:@Nadam/Nadam/update/Read/ReadVariableOp*
_output_shapes

:�
Nadam/Nadam/update/mul_5MulNadam/mul_5:z:0 Nadam/Nadam/update/truediv_1:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*9
_class/
-+loc:@Nadam/Nadam/update/Read/ReadVariableOp*
_output_shapes

:�
Nadam/Nadam/update/add_2AddV2Nadam/Nadam/update/mul_4:z:0Nadam/Nadam/update/mul_5:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*9
_class/
-+loc:@Nadam/Nadam/update/Read/ReadVariableOp*
_output_shapes

:�
Nadam/Nadam/update/mul_6MulNadam/Identity_1:output:0Nadam/Nadam/update/add_2:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*9
_class/
-+loc:@Nadam/Nadam/update/Read/ReadVariableOp*
_output_shapes

:�
Nadam/Nadam/update/SqrtSqrt Nadam/Nadam/update/truediv_2:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:*
T0*9
_class/
-+loc:@Nadam/Nadam/update/Read/ReadVariableOp�
Nadam/Nadam/update/add_3AddV2Nadam/Nadam/update/Sqrt:y:0Nadam/Const:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:*
T0*9
_class/
-+loc:@Nadam/Nadam/update/Read/ReadVariableOp�
Nadam/Nadam/update/truediv_3RealDivNadam/Nadam/update/mul_6:z:0Nadam/Nadam/update/add_3:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*9
_class/
-+loc:@Nadam/Nadam/update/Read/ReadVariableOp*
_output_shapes

:�
#Nadam/Nadam/update/ReadVariableOp_2ReadVariableOpmatmul_readvariableop_resource'^Nadam/Nadam/update/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:�
Nadam/Nadam/update/subSub+Nadam/Nadam/update/ReadVariableOp_2:value:0 Nadam/Nadam/update/truediv_3:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*9
_class/
-+loc:@Nadam/Nadam/update/Read/ReadVariableOp*
_output_shapes

:�
%Nadam/Nadam/update/AssignVariableOp_2AssignVariableOpmatmul_readvariableop_resourceNadam/Nadam/update/sub:z:0$^Nadam/Nadam/update/ReadVariableOp_2",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
 *9
_class/
-+loc:@Nadam/Nadam/update/Read/ReadVariableOp�
(Nadam/Nadam/update_1/Read/ReadVariableOpReadVariableOpadd_readvariableop_resource^add/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:�
Nadam/Nadam/update_1/IdentityIdentity0Nadam/Nadam/update_1/Read/ReadVariableOp:value:0*
_output_shapes
:*
T0�
Nadam/Nadam/update_1/truedivRealDivReshape_5:output:0Nadam/sub_5:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
T0*;
_class1
/-loc:@Nadam/Nadam/update_1/Read/ReadVariableOp�
'Nadam/Nadam/update_1/mul/ReadVariableOpReadVariableOp0nadam_nadam_update_1_mul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:�
Nadam/Nadam/update_1/mulMulNadam/Identity_2:output:0/Nadam/Nadam/update_1/mul/ReadVariableOp:value:0",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@Nadam/Nadam/update_1/Read/ReadVariableOp*
_output_shapes
:*
T0�
Nadam/Nadam/update_1/mul_1MulNadam/sub_2:z:0Reshape_5:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*;
_class1
/-loc:@Nadam/Nadam/update_1/Read/ReadVariableOp*
_output_shapes
:�
Nadam/Nadam/update_1/addAddV2Nadam/Nadam/update_1/mul:z:0Nadam/Nadam/update_1/mul_1:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*;
_class1
/-loc:@Nadam/Nadam/update_1/Read/ReadVariableOp*
_output_shapes
:�
%Nadam/Nadam/update_1/AssignVariableOpAssignVariableOp0nadam_nadam_update_1_mul_readvariableop_resourceNadam/Nadam/update_1/add:z:0(^Nadam/Nadam/update_1/mul/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
 *;
_class1
/-loc:@Nadam/Nadam/update_1/Read/ReadVariableOp�
#Nadam/Nadam/update_1/ReadVariableOpReadVariableOp0nadam_nadam_update_1_mul_readvariableop_resource&^Nadam/Nadam/update_1/AssignVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@Nadam/Nadam/update_1/Read/ReadVariableOp*
dtype0*
_output_shapes
:�
Nadam/Nadam/update_1/truediv_1RealDiv+Nadam/Nadam/update_1/ReadVariableOp:value:0Nadam/sub_6:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
T0*;
_class1
/-loc:@Nadam/Nadam/update_1/Read/ReadVariableOp�
)Nadam/Nadam/update_1/mul_2/ReadVariableOpReadVariableOp2nadam_nadam_update_1_mul_2_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:�
Nadam/Nadam/update_1/mul_2MulNadam/Identity_3:output:01Nadam/Nadam/update_1/mul_2/ReadVariableOp:value:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
T0*;
_class1
/-loc:@Nadam/Nadam/update_1/Read/ReadVariableOp�
Nadam/Nadam/update_1/SquareSquareReshape_5:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*;
_class1
/-loc:@Nadam/Nadam/update_1/Read/ReadVariableOp*
_output_shapes
:�
Nadam/Nadam/update_1/mul_3MulNadam/sub_3:z:0Nadam/Nadam/update_1/Square:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*;
_class1
/-loc:@Nadam/Nadam/update_1/Read/ReadVariableOp*
_output_shapes
:�
Nadam/Nadam/update_1/add_1AddV2Nadam/Nadam/update_1/mul_2:z:0Nadam/Nadam/update_1/mul_3:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
T0*;
_class1
/-loc:@Nadam/Nadam/update_1/Read/ReadVariableOp�
'Nadam/Nadam/update_1/AssignVariableOp_1AssignVariableOp2nadam_nadam_update_1_mul_2_readvariableop_resourceNadam/Nadam/update_1/add_1:z:0*^Nadam/Nadam/update_1/mul_2/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
 *;
_class1
/-loc:@Nadam/Nadam/update_1/Read/ReadVariableOp�
%Nadam/Nadam/update_1/ReadVariableOp_1ReadVariableOp2nadam_nadam_update_1_mul_2_readvariableop_resource(^Nadam/Nadam/update_1/AssignVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@Nadam/Nadam/update_1/Read/ReadVariableOp*
dtype0*
_output_shapes
:�
Nadam/Nadam/update_1/truediv_2RealDiv-Nadam/Nadam/update_1/ReadVariableOp_1:value:0Nadam/sub_7:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*;
_class1
/-loc:@Nadam/Nadam/update_1/Read/ReadVariableOp*
_output_shapes
:�
Nadam/Nadam/update_1/mul_4MulNadam/sub_4:z:0 Nadam/Nadam/update_1/truediv:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*;
_class1
/-loc:@Nadam/Nadam/update_1/Read/ReadVariableOp*
_output_shapes
:�
Nadam/Nadam/update_1/mul_5MulNadam/mul_5:z:0"Nadam/Nadam/update_1/truediv_1:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*;
_class1
/-loc:@Nadam/Nadam/update_1/Read/ReadVariableOp*
_output_shapes
:�
Nadam/Nadam/update_1/add_2AddV2Nadam/Nadam/update_1/mul_4:z:0Nadam/Nadam/update_1/mul_5:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
T0*;
_class1
/-loc:@Nadam/Nadam/update_1/Read/ReadVariableOp�
Nadam/Nadam/update_1/mul_6MulNadam/Identity_1:output:0Nadam/Nadam/update_1/add_2:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
T0*;
_class1
/-loc:@Nadam/Nadam/update_1/Read/ReadVariableOp�
Nadam/Nadam/update_1/SqrtSqrt"Nadam/Nadam/update_1/truediv_2:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*;
_class1
/-loc:@Nadam/Nadam/update_1/Read/ReadVariableOp*
_output_shapes
:�
Nadam/Nadam/update_1/add_3AddV2Nadam/Nadam/update_1/Sqrt:y:0Nadam/Const:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
T0*;
_class1
/-loc:@Nadam/Nadam/update_1/Read/ReadVariableOp�
Nadam/Nadam/update_1/truediv_3RealDivNadam/Nadam/update_1/mul_6:z:0Nadam/Nadam/update_1/add_3:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
T0*;
_class1
/-loc:@Nadam/Nadam/update_1/Read/ReadVariableOp�
%Nadam/Nadam/update_1/ReadVariableOp_2ReadVariableOpadd_readvariableop_resource)^Nadam/Nadam/update_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0�
Nadam/Nadam/update_1/subSub-Nadam/Nadam/update_1/ReadVariableOp_2:value:0"Nadam/Nadam/update_1/truediv_3:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*;
_class1
/-loc:@Nadam/Nadam/update_1/Read/ReadVariableOp*
_output_shapes
:�
'Nadam/Nadam/update_1/AssignVariableOp_2AssignVariableOpadd_readvariableop_resourceNadam/Nadam/update_1/sub:z:0&^Nadam/Nadam/update_1/ReadVariableOp_2",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@Nadam/Nadam/update_1/Read/ReadVariableOp*
dtype0*
_output_shapes
 �
Nadam/Nadam/ConstConst&^Nadam/Nadam/update/AssignVariableOp_2(^Nadam/Nadam/update_1/AssignVariableOp_2*
dtype0	*
_output_shapes
: *
value	B	 R�
Nadam/Nadam/AssignAddVariableOpAssignAddVariableOpnadam_readvariableop_resourceNadam/Nadam/Const:output:0^Nadam/ReadVariableOp_1*
dtype0	*
_output_shapes
 *U
_input_shapesD
B:	�N:	�N:::::::::::20
Nadam/ReadVariableOp_1Nadam/ReadVariableOp_12N
%Nadam/Nadam/update/AssignVariableOp_2%Nadam/Nadam/update/AssignVariableOp_22R
'Nadam/Nadam/update_1/mul/ReadVariableOp'Nadam/Nadam/update_1/mul/ReadVariableOp2P
&Nadam/Nadam/update/Read/ReadVariableOp&Nadam/Nadam/update/Read/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp2J
#Nadam/Nadam/update_1/ReadVariableOp#Nadam/Nadam/update_1/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2B
Nadam/Identity_4/ReadVariableOpNadam/Identity_4/ReadVariableOp20
Nadam/AssignVariableOpNadam/AssignVariableOp2,
Nadam/ReadVariableOpNadam/ReadVariableOp2:
Nadam/Cast_1/ReadVariableOpNadam/Cast_1/ReadVariableOp2J
#Nadam/Nadam/update/ReadVariableOp_1#Nadam/Nadam/update/ReadVariableOp_12J
#Nadam/Nadam/update/ReadVariableOp_2#Nadam/Nadam/update/ReadVariableOp_22N
%Nadam/Nadam/update_1/ReadVariableOp_1%Nadam/Nadam/update_1/ReadVariableOp_12N
%Nadam/Nadam/update_1/ReadVariableOp_2%Nadam/Nadam/update_1/ReadVariableOp_22R
'Nadam/Nadam/update_1/AssignVariableOp_1'Nadam/Nadam/update_1/AssignVariableOp_12N
%Nadam/Nadam/update/mul/ReadVariableOp%Nadam/Nadam/update/mul/ReadVariableOp2R
'Nadam/Nadam/update_1/AssignVariableOp_2'Nadam/Nadam/update_1/AssignVariableOp_22F
!Nadam/Nadam/update/ReadVariableOp!Nadam/Nadam/update/ReadVariableOp2:
Nadam/Cast_2/ReadVariableOpNadam/Cast_2/ReadVariableOp2R
'Nadam/Nadam/update/mul_2/ReadVariableOp'Nadam/Nadam/update/mul_2/ReadVariableOp2B
Nadam/Nadam/AssignAddVariableOpNadam/Nadam/AssignAddVariableOp2V
)Nadam/Nadam/update_1/mul_2/ReadVariableOp)Nadam/Nadam/update_1/mul_2/ReadVariableOp2J
#Nadam/Nadam/update/AssignVariableOp#Nadam/Nadam/update/AssignVariableOp2N
%Nadam/Nadam/update_1/AssignVariableOp%Nadam/Nadam/update_1/AssignVariableOp26
Nadam/Cast/ReadVariableOpNadam/Cast/ReadVariableOp2>
Nadam/Identity/ReadVariableOpNadam/Identity/ReadVariableOp2T
(Nadam/Nadam/update_1/Read/ReadVariableOp(Nadam/Nadam/update_1/Read/ReadVariableOp2N
%Nadam/Nadam/update/AssignVariableOp_1%Nadam/Nadam/update/AssignVariableOp_1:' #
!
_user_specified_name	vectors:&"
 
_user_specified_namelabels: : : : : : : :	 :
 : : 
�#
�
__inference__traced_save_1758
file_prefix 
savev2_w_read_readvariableop 
savev2_b_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop)
%savev2_nadam_iter_read_readvariableop	+
'savev2_nadam_beta_1_read_readvariableop+
'savev2_nadam_beta_2_read_readvariableop*
&savev2_nadam_decay_read_readvariableop2
.savev2_nadam_learning_rate_read_readvariableop3
/savev2_nadam_momentum_cache_read_readvariableop(
$savev2_nadam_w_m_read_readvariableop(
$savev2_nadam_b_m_read_readvariableop(
$savev2_nadam_w_v_read_readvariableop(
$savev2_nadam_b_v_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2�SaveV2_1�
StringJoin/inputs_1Const"/device:CPU:0*<
value3B1 B+_temp_fc4230da9cf14694a4ac0252e8865f29/part*
dtype0*
_output_shapes
: s

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: L

num_shardsConst*
value	B :*
dtype0*
_output_shapes
: f
ShardedFilename/shardConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*�
value�B�BW/.ATTRIBUTES/VARIABLE_VALUEBb/.ATTRIBUTES/VARIABLE_VALUEB'lrLoss/total/.ATTRIBUTES/VARIABLE_VALUEB'lrLoss/count/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB8W/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB8b/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB8W/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB8b/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:�
SaveV2/shape_and_slicesConst"/device:CPU:0*/
value&B$B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0savev2_w_read_readvariableopsavev2_b_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop%savev2_nadam_iter_read_readvariableop'savev2_nadam_beta_1_read_readvariableop'savev2_nadam_beta_2_read_readvariableop&savev2_nadam_decay_read_readvariableop.savev2_nadam_learning_rate_read_readvariableop/savev2_nadam_momentum_cache_read_readvariableop$savev2_nadam_w_m_read_readvariableop$savev2_nadam_b_m_read_readvariableop$savev2_nadam_w_v_read_readvariableop$savev2_nadam_b_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *
dtypes
2	h
ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: �
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:q
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
valueB
B *
dtype0�
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
dtypes
2*
_output_shapes
 �
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
_output_shapes
:*
T0�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
_output_shapes
: *
T0"!

identity_1Identity_1:output:0*W
_input_shapesF
D: ::: : : : : : : : ::::: 2
SaveV2SaveV22(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2_1SaveV2_1: :	 :
 : : : : : :+ '
%
_user_specified_namefile_prefix: : : : : : : "wJ
saver_filename:0StatefulPartitionedCall:0StatefulPartitionedCall_18"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:�
g
W
b

lrLoss
	optimizer

signatures

train_step"
_generic_user_object
:2W
:2b
�
	total
	count

_fn_kwargs
	regularization_losses

trainable_variables
	variables
	keras_api
*&call_and_return_all_conditional_losses
__call__"�
_tf_keras_layer�{"class_name": "MeanSquaredError", "name": "mean_squared_error", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "mean_squared_error", "dtype": "float32"}}
�
iter

beta_1

beta_2
	decay
learning_rate
momentum_cachemmvv"
	optimizer
"
signature_map
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
non_trainable_variables
	regularization_losses
layer_regularization_losses

trainable_variables
	variables
metrics

layers
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
:	 (2
Nadam/iter
: (2Nadam/beta_1
: (2Nadam/beta_2
: (2Nadam/decay
: (2Nadam/learning_rate
: 2Nadam/momentum_cache
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:2	Nadam/W/m
:2	Nadam/b/m
:2	Nadam/W/v
:2	Nadam/b/v
�2�
__inference_train_step_1694�
���
FullArgSpec(
args �
jself
	jvectors
jlabels
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 t
__inference_train_step_1694UB�?
8�5
�
vectors	�N
�
labels	�N
� "
 