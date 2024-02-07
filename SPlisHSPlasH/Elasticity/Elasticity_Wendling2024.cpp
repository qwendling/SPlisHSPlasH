#include "Elasticity_Wendling2024.h"
#include "SPlisHSPlasH/Simulation.h"
#include "SPlisHSPlasH/TimeManager.h"
#include "SPlisHSPlasH/Utilities/MathFunctions.h"
#include "Utilities/Counting.h"
#include "Utilities/Timing.h"

using namespace SPH;
using namespace GenParam;

int Elasticity_Wendling2024::ITERATIONS = -1;
int Elasticity_Wendling2024::MAX_ITERATIONS = -1;
int Elasticity_Wendling2024::MAX_ERROR = -1;
int Elasticity_Wendling2024::ALPHA = -1;

Elasticity_Wendling2024::Elasticity_Wendling2024(FluidModel *model)
    : ElasticityBase(model) {
  const unsigned int numParticles = model->numActiveParticles();
  m_restVolumes.resize(numParticles);
  m_current_to_initial_index.resize(numParticles);
  m_initial_to_current_index.resize(numParticles);
  m_initialNeighbors.resize(numParticles);
  m_L.resize(numParticles);
  m_F.resize(numParticles);
  m_CH.resize(numParticles);
  m_CD.resize(numParticles);
  m_dFdX.resize(numParticles);
  m_dX.resize(numParticles);
  m_pos_prev.resize(numParticles);

  m_lambdaH.resize(numParticles);
  m_lambdaD.resize(numParticles);
  m_dCDdF.resize(numParticles);
  m_dCHdF.resize(numParticles);

  m_iterations = 0;
  m_maxIter = 100;
  m_maxError = static_cast<Real>(1.0e-4);
  m_alpha = 0.0;

  model->addField(
      {"rest volume", FieldType::Scalar,
       [&](const unsigned int i) -> Real * { return &m_restVolumes[i]; },
       true});
  model->addField(
      {"deformation gradient", FieldType::Matrix3,
       [&](const unsigned int i) -> Real * { return &m_F[i](0, 0); }});
  model->addField(
      {"correction matrix", FieldType::Matrix3,
       [&](const unsigned int i) -> Real * { return &m_L[i](0, 0); }});
}

Elasticity_Wendling2024::~Elasticity_Wendling2024(void) {
  m_model->removeFieldByName("rest volume");
  m_model->removeFieldByName("rotation");
  m_model->removeFieldByName("stress");
  m_model->removeFieldByName("deformation gradient");
  m_model->removeFieldByName("correction matrix");
}

void Elasticity_Wendling2024::deferredInit() { initValues(); }

void Elasticity_Wendling2024::initParameters() {
  ElasticityBase::initParameters();

  ITERATIONS = createNumericParameter("elasticityIterations", "Iterations",
                                      &m_iterations);
  setGroup(ITERATIONS, "Fluid Model|Elasticity");
  setDescription(ITERATIONS, "Iterations required by the elasticity solver.");
  getParameter(ITERATIONS)->setReadOnly(true);

  MAX_ITERATIONS = createNumericParameter(
      "elasticityMaxIter", "Max. iterations (elasticity)", &m_maxIter);
  setGroup(MAX_ITERATIONS, "Fluid Model|Elasticity");
  setDescription(MAX_ITERATIONS,
                 "Coefficient for the elasticity force computation");
  static_cast<NumericParameter<unsigned int> *>(getParameter(MAX_ITERATIONS))
      ->setMinValue(1);

  MAX_ERROR = createNumericParameter("elasticityMaxError",
                                     "Max. elasticity error", &m_maxError);
  setGroup(MAX_ERROR, "Fluid Model|Elasticity");
  setDescription(MAX_ERROR, "Coefficient for the elasticity force computation");
  RealParameter *rparam = static_cast<RealParameter *>(getParameter(MAX_ERROR));
  rparam->setMinValue(1e-7);

  ALPHA = createNumericParameter("alpha", "Zero-energy modes suppression",
                                 &m_alpha);
  setGroup(ALPHA, "Fluid Model|Elasticity");
  setDescription(ALPHA, "Coefficent for zero-energy modes suppression method");
  rparam = static_cast<RealParameter *>(getParameter(ALPHA));
  rparam->setMinValue(0.0);
}

void Elasticity_Wendling2024::initValues() {
  Simulation *sim = Simulation::getCurrent();
  sim->getNeighborhoodSearch()->find_neighbors();

  FluidModel *model = m_model;
  const unsigned int numParticles = model->numActiveParticles();
  const unsigned int fluidModelIndex = model->getPointSetIndex();

// Store the neighbors in the reference configurations and
// compute the volume of each particle in rest state
#pragma omp parallel default(shared)
  {
#pragma omp for schedule(static)
    for (int i = 0; i < (int)numParticles; i++) {
      m_current_to_initial_index[i] = i;
      m_initial_to_current_index[i] = i;

      // only neighbors in same phase will influence elasticity
      const unsigned int numNeighbors =
          sim->numberOfNeighbors(fluidModelIndex, fluidModelIndex, i);
      m_initialNeighbors[i].resize(numNeighbors);
      for (unsigned int j = 0; j < numNeighbors; j++)
        m_initialNeighbors[i][j] =
            sim->getNeighbor(fluidModelIndex, fluidModelIndex, i, j);

      // compute volume
      Real density = model->getMass(i) * sim->W_zero();
      const Vector3r &xi = model->getPosition(i);
      for (size_t j = 0; j < m_initialNeighbors[i].size(); j++) {
        const unsigned int neighborIndex = m_initialNeighbors[i][j];
        const Vector3r &xj = model->getPosition(neighborIndex);
        density += model->getMass(neighborIndex) * sim->W(xi - xj);
      }
      m_restVolumes[i] = model->getMass(i) / density;
    }
  }

  // mark all particles in the bounding box as fixed
  determineFixedParticles();

  computeMatrixL();

#pragma omp parallel default(shared)
  {
#pragma omp for schedule(static)
    for (int i = 0; i < (int)numParticles; i++) {
      const unsigned int i0 = m_current_to_initial_index[i];
      const Vector3r &xi0 = m_model->getPosition0(i0);

      m_dFdX[i].setZero();

      const size_t numNeighbors = m_initialNeighbors[i0].size();

      //////////////////////////////////////////////////////////////////////////
      // Fluid
      //////////////////////////////////////////////////////////////////////////
      for (unsigned int j = 0; j < numNeighbors; j++) {
        const unsigned int neighborIndex =
            m_initial_to_current_index[m_initialNeighbors[i0][j]];
        // get initial neighbor index considering the current particle order
        const unsigned int neighborIndex0 = m_initialNeighbors[i0][j];
        const Vector3r &xj0 = m_model->getPosition0(neighborIndex0);
        const Vector3r xi_xj_0 = xi0 - xj0;
        const Vector3r correctedKernel = m_L[i] * sim->gradW(xi_xj_0);
        m_dFdX[i] += m_restVolumes[neighborIndex] * correctedKernel;
      }
    }
  }
}

void Elasticity_Wendling2024::computeConstraintH() {
  Simulation *sim = Simulation::getCurrent();
  const unsigned int numParticles = m_model->numActiveParticles();
  FluidModel *model = m_model;

  const Real dt = TimeManager::getCurrent()->getTimeStepSize();

  Real mu = m_youngsModulus / (static_cast<Real>(2.0) *
                               (static_cast<Real>(1.0) + m_poissonRatio));
  Real lambda =
      m_youngsModulus * m_poissonRatio /
      ((static_cast<Real>(1.0) + m_poissonRatio) *
       (static_cast<Real>(1.0) - static_cast<Real>(2.0) * m_poissonRatio));

#pragma omp parallel default(shared)
  {
#pragma omp for schedule(static)
    for (int i = 0; i < (int)numParticles; i++) {
      const unsigned int i0 = m_current_to_initial_index[i];
      const Vector3r &xi = m_model->getPosition(i);
      const Vector3r &xi0 = m_model->getPosition0(i0);

      Matrix3r &F = m_F[i];
      F.setZero();

      const size_t numNeighbors = m_initialNeighbors[i0].size();

      //////////////////////////////////////////////////////////////////////////
      // Fluid
      //////////////////////////////////////////////////////////////////////////
      for (unsigned int j = 0; j < numNeighbors; j++) {
        const unsigned int neighborIndex =
            m_initial_to_current_index[m_initialNeighbors[i0][j]];
        // get initial neighbor index considering the current particle order
        const unsigned int neighborIndex0 = m_initialNeighbors[i0][j];

        const Vector3r &xj = model->getPosition(neighborIndex);
        const Vector3r &xj0 = m_model->getPosition0(neighborIndex0);
        const Vector3r xj_xi = xj - xi;
        const Vector3r xi_xj_0 = xi0 - xj0;
        const Vector3r correctedKernel = m_L[i] * sim->gradW(xi_xj_0);
        F += m_restVolumes[neighborIndex] * xj_xi * correctedKernel.transpose();
      }

      if (sim->is2DSimulation())
        F(2, 2) = 1.0;

      m_CD[i] = sqrt((F.transpose() * F).trace());
      m_CH[i] = F.determinant() - (1. + mu / lambda);
      // m_CH[i] = F.determinant() - (1.);

      Matrix3r &dCDdF = m_dCDdF[i];
      dCDdF = F / m_CD[i];
      double alphaD = 1. / (mu * m_restVolumes[i]);

      Matrix3r &dCHdF = m_dCHdF[i];
      dCHdF.col(0) = F.col(1).cross(F.col(2));
      dCHdF.col(1) = F.col(2).cross(F.col(0));
      dCHdF.col(2) = F.col(0).cross(F.col(1));
      double alphaH = 1. / (lambda * m_restVolumes[i]);

      double denumH =
          1. / model->getMass(i) * (-dCHdF * m_dFdX[i0]).squaredNorm();
      double denumD =
          1. / model->getMass(i) * (-dCDdF * m_dFdX[i0]).squaredNorm();

      for (unsigned int j = 0; j < numNeighbors; j++) {
        const unsigned int neighborIndex =
            m_initial_to_current_index[m_initialNeighbors[i0][j]];
        if ((int)neighborIndex == i)
          continue;
        // get initial neighbor index considering the current particle order
        const unsigned int neighborIndex0 = m_initialNeighbors[i0][j];

        const Vector3r &xj0 = m_model->getPosition0(neighborIndex0);
        const Vector3r xi_xj_0 = xi0 - xj0;
        const Vector3r correctedKernel = m_L[i] * sim->gradW(xi_xj_0);
        denumH += 1. / model->getMass(neighborIndex) *
                  (dCHdF * m_restVolumes[neighborIndex] * correctedKernel)
                      .squaredNorm();
        denumD += 1. / model->getMass(neighborIndex) *
                  (dCDdF * m_restVolumes[neighborIndex] * correctedKernel)
                      .squaredNorm();
      }

      denumD += alphaD / (dt * dt);
      denumH += alphaH / (dt * dt);

      m_lambdaH[i] = -m_CH[i] / denumH;
      m_lambdaD[i] = -m_CD[i] / denumD;

      m_dX[i] = 1. / model->getMass(i) * -dCHdF * m_dFdX[i0] * m_lambdaH[i];
      m_dX[i] += 1. / model->getMass(i) * -dCDdF * m_dFdX[i0] * m_lambdaD[i];
    }
  }
  START_TIMING("COMPUTE_DX")
#pragma omp parallel default(shared)
  {
#pragma omp for schedule(static)
    for (int i = 0; i < (int)numParticles; i++) {
      const unsigned int i0 = m_current_to_initial_index[i];
      const Vector3r &xi0 = m_model->getPosition0(i0);

      const size_t numNeighbors = m_initialNeighbors[i0].size();
      Vector3r &dX = m_dX[i];

      for (unsigned int j = 0; j < numNeighbors; j++) {
        const unsigned int neighborIndex =
            m_initial_to_current_index[m_initialNeighbors[i0][j]];
        if ((int)neighborIndex == i)
          continue;
        // get initial neighbor index considering the current particle order
        const unsigned int neighborIndex0 = m_initialNeighbors[i0][j];

        const Vector3r &xj0 = m_model->getPosition0(neighborIndex0);
        const Vector3r xj_xi_0 = xj0 - xi0;
        const Vector3r correctedKernel =
            m_L[neighborIndex] * sim->gradW(xj_xi_0);

        Vector3r gradH =
            m_dCHdF[neighborIndex] * m_restVolumes[i] * correctedKernel;
        Vector3r gradD =
            m_dCDdF[neighborIndex] * m_restVolumes[i] * correctedKernel;
        dX += 1. / model->getMass(i) *
              (m_lambdaD[neighborIndex] * gradD +
               m_lambdaH[neighborIndex] * gradH);
      }
      dX *= 1. / (2. * numNeighbors);
    }
  }
  STOP_TIMING_AVG
}
void Elasticity_Wendling2024::computeConstraintD() {}
void Elasticity_Wendling2024::computeZeroEnergy() {}

void Elasticity_Wendling2024::step() {
  START_TIMING("Elasticity_Wendling2024")
  // apply accelerations
  const unsigned int numParticles = m_model->numActiveParticles();
  const Real dt = TimeManager::getCurrent()->getTimeStepSize();
#pragma omp parallel default(shared)
  {
#pragma omp for schedule(static)
    for (int i = 0; i < (int)numParticles; i++) {
      if (m_model->getParticleState(i) == ParticleState::Active) {
        Vector3r &vel = m_model->getVelocity(i);
        vel += dt * m_model->getAcceleration(i);
        vel.setZero();
        m_model->getAcceleration(i).setZero();
        m_pos_prev[i] = m_model->getPosition(i);
        Vector3r &pos = m_model->getPosition(i);
        pos += dt * vel;
      }
    }
  }

  START_TIMING("Elasticity")

  computeConstraintH();
  // computeConstraintD();
  // computeZeroEnergy();

  STOP_TIMING_AVG
#pragma omp parallel default(shared)
  {
#pragma omp for schedule(static)
    for (int i = 0; i < (int)numParticles; i++) {
      if (m_model->getParticleState(i) == ParticleState::Active) {
        Vector3r &pos = m_model->getPosition(i);
        pos += m_dX[i];

        Vector3r &vel = m_model->getVelocity(i);
        vel = (pos - m_pos_prev[i]) / dt;

        pos = m_pos_prev[i];
      }
    }
  }
  STOP_TIMING_AVG
}

void Elasticity_Wendling2024::reset() { initValues(); }

void Elasticity_Wendling2024::performNeighborhoodSearchSort() {
  const unsigned int numPart = m_model->numActiveParticles();
  if (numPart == 0)
    return;

  Simulation *sim = Simulation::getCurrent();
  auto const &d =
      sim->getNeighborhoodSearch()->point_set(m_model->getPointSetIndex());
  d.sort_field(&m_restVolumes[0]);
  d.sort_field(&m_current_to_initial_index[0]);
  d.sort_field(&m_L[0]);

  for (unsigned int i = 0; i < numPart; i++)
    m_initial_to_current_index[m_current_to_initial_index[i]] = i;
}

void Elasticity_Wendling2024::computeMatrixL() {
  Simulation *sim = Simulation::getCurrent();
  const unsigned int numParticles = m_model->numActiveParticles();
  const unsigned int fluidModelIndex = m_model->getPointSetIndex();

#pragma omp parallel default(shared)
  {
#pragma omp for schedule(static)
    for (int i = 0; i < (int)numParticles; i++) {
      const unsigned int i0 = m_current_to_initial_index[i];
      const Vector3r &xi0 = m_model->getPosition0(i0);
      Matrix3r L;
      L.setZero();

      const size_t numNeighbors = m_initialNeighbors[i0].size();

      //////////////////////////////////////////////////////////////////////////
      // Fluid
      //////////////////////////////////////////////////////////////////////////
      for (unsigned int j = 0; j < numNeighbors; j++) {
        const unsigned int neighborIndex =
            m_initial_to_current_index[m_initialNeighbors[i0][j]];
        // get initial neighbor index considering the current particle order
        const unsigned int neighborIndex0 = m_initialNeighbors[i0][j];

        const Vector3r &xj0 = m_model->getPosition0(neighborIndex0);
        const Vector3r xj_xi_0 = xj0 - xi0;
        const Vector3r gradW = sim->gradW(xj_xi_0);

        // minus because gradW(xij0) == -gradW(xji0)
        L -= m_restVolumes[neighborIndex] * gradW * xj_xi_0.transpose();
      }

      // add 1 to z-component. otherwise we get a singular matrix in 2D
      if (sim->is2DSimulation())
        L(2, 2) = 1.0;

      bool invertible = false;
      L.computeInverseWithCheck(m_L[i], invertible, 1e-9);
      if (!invertible) {
        // MathFunctions::pseudoInverse(L, m_L[i]);
        m_L[i] = Matrix3r::Zero();
      }
    }
  }
}

void SPH::Elasticity_Wendling2024::saveState(BinaryFileWriter &binWriter) {
  binWriter.writeBuffer((char *)m_current_to_initial_index.data(),
                        m_current_to_initial_index.size() *
                            sizeof(unsigned int));
  binWriter.writeBuffer((char *)m_initial_to_current_index.data(),
                        m_initial_to_current_index.size() *
                            sizeof(unsigned int));
  binWriter.writeBuffer((char *)m_L.data(), m_L.size() * sizeof(Matrix3r));
}

void SPH::Elasticity_Wendling2024::loadState(BinaryFileReader &binReader) {
  binReader.readBuffer((char *)m_current_to_initial_index.data(),
                       m_current_to_initial_index.size() *
                           sizeof(unsigned int));
  binReader.readBuffer((char *)m_initial_to_current_index.data(),
                       m_initial_to_current_index.size() *
                           sizeof(unsigned int));
  binReader.readBuffer((char *)m_L.data(), m_L.size() * sizeof(Matrix3r));
}
