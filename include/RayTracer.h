#ifndef __RayTracer_h
#define __RayTracer_h

//[]------------------------------------------------------------------------[]
//|                                                                          |
//|                          GVSG Graphics Library                           |
//|                               Version 1.0                                |
//|                                                                          |
//|              Copyright� 2007-2016, Paulo Aristarco Pagliosa              |
//|              All Rights Reserved.                                        |
//|                                                                          |
//[]------------------------------------------------------------------------[]
//
//  OVERVIEW: RayTracer.h
//  ========
//  Class definition for simple ray tracer.

#include "Image.h"
#include "Intersection.h"
#include "Renderer.h"

namespace Graphics
{ // begin namespace Graphics

#define MIN_WEIGHT REAL(0.001)
#define MAX_RECURSION_LEVEL 20


//////////////////////////////////////////////////////////
//
// RayTracer: simple ray tracer class
// =========
class RayTracer: public Renderer
{
public:
  struct DebugInfo
  {
    Ray ray;
    Intersection hit;

  };

  // Constructor
  RayTracer(Scene&, Camera* = 0);

  uint getMaxRecursionLevel() const
  {
    return maxRecursionLevel;
  }

  REAL getMinWeight() const
  {
    return minWeight;
  }

  void setMaxRecursionLevel(uint rl)
  {
    maxRecursionLevel = dMin<uint>(rl, MAX_RECURSION_LEVEL);
  }

  void setMinWeight(REAL w)
  {
    minWeight = dMax<REAL>(w, MIN_WEIGHT);
  }

  void render();
  virtual void renderImage(Image&);

  void debug(int, int, DebugInfo&);

protected:
  ObjectPtr<Model> aggregate;
  uint maxRecursionLevel;
  REAL minWeight;

  virtual void scan(Image&);
  virtual void setPixelRay(REAL, REAL);
  virtual Color shoot(REAL, REAL);
  virtual Color trace(const Ray&, uint, REAL);

  // TODO: INSERT YOUR CODE HERE

}; // RayTracer

} // end namespace Graphics

#endif // __RayTracer_h
