#pragma once
#include "01_supervised/02_classification/11_lda/Model.hpp"
namespace course {
class QDA : public LDA {
public:
  explicit QDA(double regularization = 1e-3) : LDA(regularization, true) {}
};
} // namespace course
