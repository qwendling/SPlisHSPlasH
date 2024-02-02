#ifndef __Elasticity_Wendling2024_h__
#define __Elasticity_Wendling2024_h__

#include "ElasticityBase.h"
#include "SPlisHSPlasH/Common.h"
#include "SPlisHSPlasH/FluidModel.h"
#include "SPlisHSPlasH/Utilities/MatrixFreeSolver.h"

namespace SPH {
/** \brief This class implements SPH in the XPBD framework
 */
class Elasticity_Wendling2024 : public ElasticityBase {
protected:
  // initial particle indices, used to access their original positions
  std::vector<unsigned int> m_current_to_initial_index;
  std::vector<unsigned int> m_initial_to_current_index;
  // initial particle neighborhood
  std::vector<std::vector<unsigned int>> m_initialNeighbors;
  // volumes in rest configuration
  std::vector<Real> m_restVolumes;
  std::vector<Matrix3r> m_L;
  std::vector<Matrix3r> m_F;
  std::vector<double> m_CD;
  std::vector<double> m_CH;

  unsigned int m_iterations;
  unsigned int m_maxIter;
  Real m_maxError;
  Real m_alpha;

  void initValues();
  void computeMatrixL();
  void computeConstraintH();
  void computeConstraintD();
  void computeZeroEnergy();

  virtual void initParameters();
  /** This function is called after the simulation scene is loaded and all
   * parameters are initialized. While reading a scene file several parameters
   * can change. The deferred init function should initialize all values which
   * depend on these parameters.
   */
  virtual void deferredInit();

public:
  static int ITERATIONS;
  static int MAX_ITERATIONS;
  static int MAX_ERROR;
  static int ALPHA;

  Elasticity_Wendling2024(FluidModel *model);
  virtual ~Elasticity_Wendling2024(void);

  static NonPressureForceBase *creator(FluidModel *model) {
    return new Elasticity_Wendling2024(model);
  }

  virtual void step();
  virtual void reset();
  virtual void performNeighborhoodSearchSort();

  virtual void saveState(BinaryFileWriter &binWriter);
  virtual void loadState(BinaryFileReader &binReader);

  static void matrixVecProd(const Real *vec, Real *result, void *userData);
};
} // namespace SPH

#endif
