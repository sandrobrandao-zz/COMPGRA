//[]------------------------------------------------------------------------[]
//|                                                                          |
//|                        GVSG Foundation Classes                           |
//|                               Version 1.0                                |
//|                                                                          |
//|              Copyright® 2007-2016, Paulo Aristarco Pagliosa              |
//|              All Rights Reserved.                                        |
//|                                                                          |
//[]------------------------------------------------------------------------[]
//
//  OVERVIEW: Camera.cpp
//  ========
//  Source file for camera.

#include "Camera.h"
#include "Exception.h"

using namespace Graphics;

//
// Auxiiary function
//
inline void
error(const char* msg)
{
  throw Exception(msg);
}


//////////////////////////////////////////////////////////
//
// Camera implementation
// ======
uint Camera::nextId;

inline string
Camera::defaultName()
{
  char name[16];

  sprintf(name, "camera%d", ++nextId);
  return string(name);
}

inline void
Camera::updateFocalPoint()
{
  focalPoint = position + directionOfProjection * distance;
  viewModified = true;
}

inline void
Camera::updateDOP()
{
  directionOfProjection = (focalPoint - position) * (REAL)(1 / distance);
  viewModified = true;
}

void
Camera::checkViewUp(const vec3& value)
{
  if (value.isNull())
    error("View up cannot be null");
  if (directionOfProjection.cross(value).isNull())
    error("View up cannot be parallel to DOP");
}

Camera::Camera(
  const vec3& position,
  const vec3& dop,
  const vec3& viewUp,
  REAL angle,
  REAL aspect):
  NameableObject{ defaultName() },
  timestamp{ 0 },
  projectionType{ Perspective }
//[]---------------------------------------------------[]
//|  Constructor                                        |
//[]---------------------------------------------------[]
{
  distance = dop.length();
  if (distance < MIN_DISTANCE)
    distance = MIN_DISTANCE;
  directionOfProjection = dop * (REAL)(1 / distance);
  checkViewUp(viewUp);
  this->viewUp = viewUp.versor();
  this->position = position;
  focalPoint = position + dop;
  aspectRatio = aspect < MIN_ASPECT ? MIN_ASPECT : aspect;
  if (angle < MIN_ANGLE)
    angle = MIN_ANGLE;
  else if (angle > MAX_ANGLE)
    angle = MAX_ANGLE;
  F = (REAL)0.01;
  B = (REAL)1000;
  height = 2 * F * (REAL)tan(Math::toRadians(viewAngle = angle) * 0.5);
  viewModified = true;
}

void
Camera::setPosition(const vec3& value)
//[]---------------------------------------------------[]
//|  Set the camera's position                          |
//|                                                     |
//|  Setting the camera's position will not change      |
//|  neither the direction of projection nor the        |
//|  distance between the position and the focal point. |
//|  The focal point will be moved along the direction  |
//|  of projection.                                     |
//[]---------------------------------------------------[]
{
  if (position != value)
  {
    position = value;
    updateFocalPoint();
  }
}

void
Camera::setDirectionOfProjection(const vec3& value)
//[]---------------------------------------------------[]
//|  Set the direction of projection                    |
//|                                                     |
//|  Setting the direction of projection will not       |
//|  change the distance between the position and the   |
//|  focal point. The focal point will be moved along   |
//|  the direction of projection.                       |
//[]---------------------------------------------------[]
{
  if (value.isNull())
    error("Direction of projection cannot be null");

  const vec3 dop = value.versor();

  if (directionOfProjection != dop)
  {
    directionOfProjection = dop;
    updateFocalPoint();
  }
}

void
Camera::setViewUp(const vec3& value)
//[]---------------------------------------------------[]
//|  Set the camera's view up                           |
//[]---------------------------------------------------[]
{
  checkViewUp(value);

  const vec3 vup = value.versor();

  if (viewUp != vup)
  {
    viewUp = vup;
    viewModified = true;
  }
}

void
Camera::setProjectionType(ProjectionType value)
//[]---------------------------------------------------[]
//|  Set the camera's projection type                   |
//[]---------------------------------------------------[]
{
  if (projectionType != value)
  {
    projectionType = value;
    viewModified = true;
  }
}

void
Camera::setDistance(REAL value)
//[]---------------------------------------------------[]
//|  Set the camera's distance                          |
//|                                                     |
//|  Setting the distance between the position and      |
//|  focal point will move the focal point along the    |
//|  direction of projection.                           |
//[]---------------------------------------------------[]
{
  if (value <= 0)
    error("Distance must be positive");
  if (!Math::isEqual(distance, value))
  {
    distance = dMax(value, MIN_DISTANCE);
    updateFocalPoint();
  }
}

void
Camera::setViewAngle(REAL value)
//[]---------------------------------------------------[]
//|  Set the camera's view angle                        |
//[]---------------------------------------------------[]
{
  if (value <= 0)
    error("View angle must be positive");
  if (!Math::isEqual(viewAngle, value))
  {
    viewAngle = dMin(dMax(value, MIN_ANGLE), MAX_ANGLE);
    if (projectionType == Perspective)
      viewModified = true;
  }
}

void
Camera::setHeight(REAL value)
//[]---------------------------------------------------[]
//|  Set the camera's view height                       |
//[]---------------------------------------------------[]
{
  if (value <= 0)
    error("Height of the view window must be positive");
  if (!Math::isEqual(height, value))
  {
    height = dMax(value, MIN_HEIGHT);
    if (projectionType == Parallel)
      viewModified = true;
  }
}

void
Camera::setAspectRatio(REAL value)
//[]---------------------------------------------------[]
//|  Set the camera's aspect ratio                      |
//[]---------------------------------------------------[]
{
  if (value <= 0)
    error("Aspect ratio must be positive");
  if (!Math::isEqual(aspectRatio, value))
  {
    aspectRatio = dMax(value, MIN_ASPECT);
    viewModified = true;
  }
}

void
Camera::setClippingPlanes(REAL F, REAL B)
//[]---------------------------------------------------[]
//|  Set the distance of the clippling planes           |
//[]---------------------------------------------------[]
{
  if (F <= 0 || B <= 0)
    error("Clipping plane distance must be positive");
  if (F > B)
  {
    REAL temp = F;

    F = B;
    B = temp;
  }
  if (F < MIN_FRONT_PLANE)
    F = MIN_FRONT_PLANE;
  if ((B - F) < MIN_DEPTH)
    B = F + MIN_DEPTH;
  if (!Math::isEqual(this->F, F) || !Math::isEqual(this->B, B))
  {
    this->F = F;
    this->B = B;
    viewModified = true;
  }
}

void
Camera::setNearPlane(REAL F)
//[]---------------------------------------------------[]
//|  Set the distance of the near clipping plane        |
//[]---------------------------------------------------[]
{
  if (F > MIN_FRONT_PLANE && B - F > MIN_DEPTH && !Math::isEqual(this->F, F))
  {
    this->F = F;
    viewModified = true;
  }
}

inline void
transform(vec3& p, const mat4& m)
{
  p = m.transform3x4(p);
}

inline void
transformDirection(vec3& v, const mat4& m)
{
  v = m.transformVector(v).versor();
}

void
Camera::azimuth(REAL angle)
//[]---------------------------------------------------[]
//|  Azimuth                                            |
//|                                                     |
//|  Rotate the camera's position about the view up     |
//|  vector centered at the focal point.                |
//[]---------------------------------------------------[]
{
  if (!Math::isZero(angle))
  {
    const mat4 r = mat4::rotation(viewUp, angle, focalPoint);

    transform(position, r);
    updateDOP();
  }
}

void
Camera::elevation(REAL angle)
//[]---------------------------------------------------[]
//|  Elevation                                          |
//|                                                     |
//|  Rotate the camera's position about the cross       |
//|  product of the view plane normal and the view up   |
//|  vector centered at the focal point.                |
//[]---------------------------------------------------[]
{
  if (!Math::isZero(angle))
  {
    const vec3 u = directionOfProjection.cross(viewUp);
    const mat4 r = mat4::rotation(u, angle, focalPoint);

    transform(position, r);
    updateDOP();
    viewUp = u.cross(directionOfProjection);
  }
}

void
Camera::roll(REAL angle)
//[]---------------------------------------------------[]
//|  Roll                                               |
//|                                                     |
//|  Rotate the view up vector around the view plane    |
//|  normal.                                            |
//[]---------------------------------------------------[]
{
  if (!Math::isZero(angle))
  {
    const quat q(-angle, directionOfProjection);

    viewUp = q.rotate(viewUp).versor();
    viewModified = true;
  }
}

void
Camera::yaw(REAL angle)
//[]---------------------------------------------------[]
//|  Yaw                                                |
//|                                                     |
//|  Rotate the focal point about the view up vector    |
//|  centered at the camera's position.                 |
//[]---------------------------------------------------[]
{
  if (!Math::isZero(angle))
  {
    const mat4 r = mat4::rotation(viewUp, angle, position);

    transform(focalPoint, r);
    updateDOP();
  }
}

void
Camera::pitch(REAL angle)
//[]---------------------------------------------------[]
//|  Pitch                                              |
//|                                                     |
//|  Rotate the focal point about the cross product of  |
//|  the view up vector and the view plane normal       |
//|  centered at the camera's position.                 |
//[]---------------------------------------------------[]
{
  if (!Math::isZero(angle))
  {
    const vec3 u = directionOfProjection.cross(viewUp);
    const mat4 r = mat4::rotation(u, angle, position);

    transform(focalPoint, r);
    updateDOP();
    viewUp = u.cross(directionOfProjection);
  }
}

void
Camera::rotateYX(REAL ay, REAL ax)
//[]---------------------------------------------------[]
//|  Rotate YX                                          |
//|                                                     |
//|  Composition of an azimuth of ay with an elevation  |
//|  of ax.                                             |
//[]---------------------------------------------------[]
{
  const quat qy(ay, viewUp);
  vec3 u = directionOfProjection.cross(viewUp);

  u = qy.rotate(u).versor();
  
  const quat qx(ax, u);

  viewUp = qx.rotate(viewUp).versor();
  directionOfProjection = viewUp.cross(u).versor();
  position = focalPoint - directionOfProjection * distance;
  viewModified = true;
}

void
Camera::zoom(REAL zoom)
//[]---------------------------------------------------[]
//|  Zoom                                               |
//|                                                     |
//|  Change the view angle (or height) of the camera so |
//|  that more or less of a scene occupies the view     |
//|  window.  A value > 1 is a zoom-in. A value < 1 is  |
//|  zoom-out.                                          |
//[]---------------------------------------------------[]
{
  if (zoom > 0)
    if (projectionType == Perspective)
      setViewAngle(viewAngle / zoom);
    else
      setHeight(height / zoom);
}

void
Camera::move(REAL dx, REAL dy, REAL dz)
//[]---------------------------------------------------[]
//|  Move the camera                                    |
//[]---------------------------------------------------[]
{
  if (!Math::isZero(dx))
    position += directionOfProjection.cross(viewUp) * dx;
  if (!Math::isZero(dy))
    position += viewUp * dy;
  if (!Math::isZero(dz))
    position -= directionOfProjection * dz;
  updateFocalPoint();
}

void
Camera::setDefaultView()
//[]---------------------------------------------------[]
//|  Set default view                                   |
//[]---------------------------------------------------[]
{
  position.set(0, 0, 10);
  directionOfProjection.set(0, 0, -1);
  focalPoint.set(0, 0, 0);
  distance = 10;
  viewUp.set(0, 1, 0);
  aspectRatio = 1;
  projectionType = Perspective;
  viewAngle = 60;
  F = (REAL)0.01;
  B = (REAL)1000;
  height = (REAL)(0.02 / M_SQRT3); // 2 * F * tan(viewAngle / 2)
  viewModified = true;
}

inline vec3
vAxis(const vec3& DOP, const vec3& VUP)
{
  return DOP.cross(VUP).versor().cross(DOP);
}

uint
Camera::updateView()
//[]---------------------------------------------------[]
//|  Update matrix                                      |
//[]---------------------------------------------------[]
{
  if (viewModified)
  {
    if (projectionType == Parallel)
    {
      REAL t = height * REAL(0.5);
      REAL r = t * aspectRatio;

      projectionMatrix = mat4::ortho(-r, r, -t, t, F, B);
    }
    else
      projectionMatrix = mat4::perspective(viewAngle, aspectRatio, F, B);
    viewUp = vAxis(directionOfProjection, viewUp);
    matrix = mat4::lookAt(position, focalPoint, viewUp);
    matrix.inverse(inverseMatrix);
    viewModified = false;
    timestamp++;
  }
  return timestamp;
}

inline const char*
Camera::getProjectionName() const
//[]---------------------------------------------------[]
//|  Projection name                                    |
//[]---------------------------------------------------[]
{
  static const char* projectionName[] = { "Parallel", "Perspective" };
  return projectionName[projectionType];
}

void
Camera::print(FILE* f) const
//[]---------------------------------------------------[]
//|  Print camera                                       |
//[]---------------------------------------------------[]
{
  fprintf(f, "Camera name: \"%s\"\n", getName().c_str());
  fprintf(f, "Projection type: %s\n", getProjectionName());
  position.print("Position: ", f);
  directionOfProjection.print("Direction of projection: ", f);
  fprintf(f, "Distance: %f\n", distance);
  viewUp.print("View up: ", f);
  fprintf(f, "View angle/height: %f/%f\n", viewAngle, height);
}
