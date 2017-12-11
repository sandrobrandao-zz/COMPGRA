#ifndef __Model_h
#define __Model_h

//[]------------------------------------------------------------------------[]
//|                                                                          |
//|                          GVSG Graphics Library                           |
//|                               Version 1.0                                |
//|                                                                          |
//|              Copyright® 2010-2016, Paulo Aristarco Pagliosa              |
//|              All Rights Reserved.                                        |
//|                                                                          |
//[]------------------------------------------------------------------------[]
//
//  OVERVIEW: Model.h
//  ========
//  Class definition for generic model.

#include "Geometry/Bounds3.h"
#include "Intersection.h"
#include "Material.h"

using namespace Ds;

namespace Graphics
{ // begin namespace Graphics

class Model;
class TriangleMesh;

typedef ObjectPtr<Model> ModelPtr;


//////////////////////////////////////////////////////////
//
// Model: generic model class
// =====
class Model: public Object
{
public:
  // Destructor
  virtual ~Model();

  virtual bool canIntersect() const;
  virtual Array<ModelPtr> refine() const;
  virtual const TriangleMesh* triangleMesh() const;
  virtual bool intersect(const Ray&, Intersection&) const = 0;
  virtual vec3 normal(const Intersection&) const = 0;
  virtual const Material* getMaterial() const = 0;
  virtual const mat4& getLocalToWorldMatrix() const;
  virtual const mat4& getWorldToLocalMatrix() const;
  virtual Bounds3 boundingBox() const = 0;

}; // Model


//////////////////////////////////////////////////////////
//
// Aggregate: generic aggregate model class
// =========
class Aggregate: public Model
{
public:
  vec3 normal(const Intersection&) const override;
  const Material* getMaterial() const override;

}; // Aggregate


//////////////////////////////////////////////////////////
//
// Primitive: generic primitive model class
// =========
class Primitive: public Model
{
public:
  const Material* getMaterial() const override;
  const mat4& getLocalToWorldMatrix() const override;
  const mat4& getWorldToLocalMatrix() const override;

  const mat3& getNormalMatrix() const
  {
    return normalMatrix;
  }

  virtual void setMaterial(Material*);
  virtual void setTransform(const vec3&, const quat&, const vec3&);

protected:
  ObjectPtr<Material> material;
  mat4 localToWorld;
  mat4 worldToLocal;
  mat3 normalMatrix;

  // Protected constructor
  Primitive():
    material(Material::getDefault()),
    normalMatrix(mat3::identity())
  {
    localToWorld = worldToLocal = mat4::identity();
  }

  Primitive(const Primitive& p):
    material(p.material),
    localToWorld(p.localToWorld),
    worldToLocal(p.worldToLocal),
    normalMatrix(p.normalMatrix)
  {
    // do nothing
  }

}; // Primitive


//////////////////////////////////////////////////////////
//
// ModelInstance: model instance class
// =============
class ModelInstance: public Primitive
{
public:
  // Constructor
  ModelInstance(Model& m, const Primitive& p):
    Primitive(p),
    model(&m)
  {
    // do nothing
  }

  const TriangleMesh* triangleMesh() const override;
  bool intersect(const Ray&, Intersection&) const override;
  vec3 normal(const Intersection&) const override;
  Bounds3 boundingBox() const override;

private:
  ModelPtr model;

}; // ModelInstance

} // end namespace Graphics

#endif // __Model_h
