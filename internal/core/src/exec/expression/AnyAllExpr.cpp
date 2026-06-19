// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "AnyAllExpr.h"

#include <string>
#include <string_view>

#include "common/Array.h"
#include "common/EasyAssert.h"
#include "exec/expression/EvalCtx.h"
#include "exec/expression/Utils.h"
#include "pb/plan.pb.h"

namespace milvus {
namespace exec {

// applyOp returns true if (elem op val) where op is the proto OpType.
template <typename T>
static bool
applyOp(const T& elem, const T& val, proto::plan::OpType op) {
    switch (op) {
        case proto::plan::OpType::LessThan:
            return elem < val;
        case proto::plan::OpType::LessEqual:
            return elem <= val;
        case proto::plan::OpType::GreaterThan:
            return elem > val;
        case proto::plan::OpType::GreaterEqual:
            return elem >= val;
        case proto::plan::OpType::Equal:
            return elem == val;
        case proto::plan::OpType::NotEqual:
            return elem != val;
        default:
            ThrowInfo(OpTypeInvalid,
                      "unsupported op type for ANY/ALL: {}",
                      static_cast<int>(op));
    }
}

// Extract a typed scalar from GenericValue.
template <typename T>
static T
extractScalar(const proto::plan::GenericValue& gv);

template <>
bool
extractScalar<bool>(const proto::plan::GenericValue& gv) {
    AssertInfo(gv.has_bool_val(), "AnyAllExpr: expected bool value");
    return gv.bool_val();
}

template <>
int8_t
extractScalar<int8_t>(const proto::plan::GenericValue& gv) {
    AssertInfo(gv.has_int64_val(), "AnyAllExpr: expected int value");
    return static_cast<int8_t>(gv.int64_val());
}

template <>
int16_t
extractScalar<int16_t>(const proto::plan::GenericValue& gv) {
    AssertInfo(gv.has_int64_val(), "AnyAllExpr: expected int value");
    return static_cast<int16_t>(gv.int64_val());
}

template <>
int32_t
extractScalar<int32_t>(const proto::plan::GenericValue& gv) {
    AssertInfo(gv.has_int64_val(), "AnyAllExpr: expected int value");
    return static_cast<int32_t>(gv.int64_val());
}

template <>
int64_t
extractScalar<int64_t>(const proto::plan::GenericValue& gv) {
    AssertInfo(gv.has_int64_val(), "AnyAllExpr: expected int value");
    return gv.int64_val();
}

template <>
float
extractScalar<float>(const proto::plan::GenericValue& gv) {
    if (gv.has_float_val())
        return static_cast<float>(gv.float_val());
    AssertInfo(gv.has_int64_val(), "AnyAllExpr: expected numeric value");
    return static_cast<float>(gv.int64_val());
}

template <>
double
extractScalar<double>(const proto::plan::GenericValue& gv) {
    if (gv.has_float_val())
        return gv.float_val();
    AssertInfo(gv.has_int64_val(), "AnyAllExpr: expected numeric value");
    return static_cast<double>(gv.int64_val());
}

template <>
std::string
extractScalar<std::string>(const proto::plan::GenericValue& gv) {
    AssertInfo(gv.has_string_val(), "AnyAllExpr: expected string value");
    return gv.string_val();
}

template <typename ElementT>
VectorPtr
PhyAnyAllFilterExpr::ExecAnyAll(EvalCtx& context) {
    using GetType =
        std::conditional_t<std::is_same_v<ElementT, std::string>,
                           std::string_view,
                           ElementT>;

    auto* input = context.get_offset_input();
    const auto& bitmap_input = context.get_bitmap_input();
    auto real_batch_size =
        has_offset_input_ ? input->size() : GetNextBatchSize();
    if (real_batch_size == 0) {
        return nullptr;
    }

    auto res_vec =
        std::make_shared<ColumnVector>(TargetBitmap(real_batch_size, false),
                                       TargetBitmap(real_batch_size, true));
    TargetBitmapView res(res_vec->GetRawData(), real_batch_size);
    TargetBitmapView valid_res(res_vec->GetValidRawData(), real_batch_size);

    auto op = expr_->op_type_;
    bool is_any = expr_->is_any_;
    ElementT scalar_val = extractScalar<ElementT>(expr_->val_);

    int processed_cursor = 0;
    auto execute_sub_batch =
        [&processed_cursor, &bitmap_input, op, is_any, &scalar_val]<
            FilterType filter_type = FilterType::sequential>(
            const milvus::ArrayView* data,
            const bool* valid_data,
            const int32_t* offsets,
            const int size,
            TargetBitmapView res,
            TargetBitmapView valid_res) {
        if (data == nullptr) {
            processed_cursor += size;
            return;
        }
        bool has_bitmap_input = !bitmap_input.empty();
        for (int i = 0; i < size; ++i) {
            auto offset = i;
            if constexpr (filter_type == FilterType::random) {
                offset = (offsets) ? offsets[i] : i;
            }
            if (valid_data != nullptr && !valid_data[offset]) {
                res[i] = valid_res[i] = false;
                continue;
            }
            if (has_bitmap_input && !bitmap_input[processed_cursor + i]) {
                continue;
            }
            const milvus::ArrayView& arr = data[offset];
            // ANY starts false; ALL starts true.
            bool row_result = !is_any;
            for (int64_t j = 0; j < arr.length(); j++) {
                GetType elem = arr.get_data<GetType>(j);
                bool match = applyOp<ElementT>(ElementT(elem), scalar_val, op);
                if (is_any && match) {
                    row_result = true;
                    break;
                }
                if (!is_any && !match) {
                    row_result = false;
                    break;
                }
            }
            res[i] = row_result;
        }
        processed_cursor += size;
    };

    int64_t processed_size;
    if (has_offset_input_) {
        processed_size = ProcessDataByOffsets<milvus::ArrayView>(
            execute_sub_batch, std::nullptr_t{}, input, res, valid_res);
    } else {
        processed_size = ProcessDataChunks<milvus::ArrayView>(
            execute_sub_batch, std::nullptr_t{}, res, valid_res);
    }
    AssertInfo(processed_size == real_batch_size,
               "internal error: AnyAllExpr processed rows {} not equal "
               "expected batch size {}",
               processed_size,
               real_batch_size);
    return res_vec;
}

void
PhyAnyAllFilterExpr::Eval(EvalCtx& context, VectorPtr& result) {
    auto element_type =
        static_cast<DataType>(expr_->column_.element_type_);

    switch (element_type) {
        case DataType::BOOL:
            result = ExecAnyAll<bool>(context);
            break;
        case DataType::INT8:
            result = ExecAnyAll<int8_t>(context);
            break;
        case DataType::INT16:
            result = ExecAnyAll<int16_t>(context);
            break;
        case DataType::INT32:
            result = ExecAnyAll<int32_t>(context);
            break;
        case DataType::INT64:
            result = ExecAnyAll<int64_t>(context);
            break;
        case DataType::FLOAT:
            result = ExecAnyAll<float>(context);
            break;
        case DataType::DOUBLE:
            result = ExecAnyAll<double>(context);
            break;
        case DataType::VARCHAR:
        case DataType::STRING:
            result = ExecAnyAll<std::string>(context);
            break;
        default:
            ThrowInfo(DataTypeInvalid,
                      "AnyAllExpr: unsupported array element type: {}",
                      static_cast<int>(element_type));
    }
}

}  // namespace exec
}  // namespace milvus
