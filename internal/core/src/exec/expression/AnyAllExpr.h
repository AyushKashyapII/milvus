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

#pragma once

#include <fmt/core.h>
#include <string>

#include "common/EasyAssert.h"
#include "common/Types.h"
#include "exec/expression/Expr.h"
#include "expr/ITypeExpr.h"
#include "segcore/SegmentInterface.h"

namespace milvus {
namespace exec {

class PhyAnyAllFilterExpr : public SegmentExpr {
 public:
    PhyAnyAllFilterExpr(
        const std::vector<std::shared_ptr<Expr>>& input,
        const std::shared_ptr<const milvus::expr::AnyAllFilterExpr>& expr,
        const std::string& name,
        milvus::OpContext* op_ctx,
        const segcore::SegmentInternalInterface* segment,
        int64_t active_count,
        milvus::Timestamp timestamp,
        int64_t batch_size,
        int32_t consistency_level)
        : SegmentExpr(std::move(input),
                      name,
                      op_ctx,
                      segment,
                      expr->column_.field_id_,
                      expr->column_.nested_path_,
                      DataType::ARRAY,
                      active_count,
                      batch_size,
                      consistency_level),
          expr_(expr),
          timestamp_(timestamp) {
    }

    void
    Eval(EvalCtx& context, VectorPtr& result) override;

    bool
    IsSource() const override {
        return true;
    }

    std::string
    ToString() const override {
        return fmt::format("{}", expr_->ToString());
    }

    std::optional<milvus::expr::ColumnInfo>
    GetColumnInfo() const override {
        return expr_->column_;
    }

 private:
    template <typename ElementT>
    VectorPtr
    ExecAnyAll(EvalCtx& context);

    const std::shared_ptr<const milvus::expr::AnyAllFilterExpr> expr_;
    milvus::Timestamp timestamp_;
};

}  // namespace exec
}  // namespace milvus
