#include <GL/glew.h>
#include <GL/freeglut.h>
#include <stdio.h>

#include "Camera.h"
#include "GLProgram.h"

using namespace std;
using namespace Graphics;

#define WIN_W 640
#define WIN_H 480

// Rendering program
GLSL::Program* program;

// Rendering func
void (*render)();

// Camera globals
const float CAMERA_RES = 360.f;
const float ZOOM_SCALE = 1.01f;
Camera* camera;
int W;
int H;

// Keyboard globals
const int MAX_KEYS = 256;
bool keys[MAX_KEYS];

// Mouse globals
int mouseX;
int mouseY;

// Scene globals
int useVertexColors;
int shadingMode; // 0: Gouraud, 1: Phong

inline mat3f
normalMatrix(const mat4f& trs, const Camera* c)
{
  mat3f m(trs);

  m[0] *= 1 / m[0].normSquared();
  m[1] *= 1 / m[1].normSquared();
  m[2] *= 1 / m[2].normSquared();
  return mat3(c->getWorldToCameraMatrix()) * m;
}

template <typename T>
inline int
size(const T* p, int n)
{
  return sizeof(T) * n;
}

// Triangle indices
struct Triangle
{
  int v[3];
};

class GLMeshArray
{
private:
  GLuint vao;
  GLuint buffers[4];
  int count;

public:
  GLMeshArray(
    const vec4f* v,
    const vec3f* n,
    const Color* c,
    int nv,
    const Triangle* t,
    int nt)
  {
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(4, buffers);
    glBindBuffer(GL_ARRAY_BUFFER, buffers[0]);
    glBufferData(GL_ARRAY_BUFFER, size(v, nv), v, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, buffers[1]);
    glBufferData(GL_ARRAY_BUFFER, size(n, nv), n, GL_STATIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, buffers[2]);
    glBufferData(GL_ARRAY_BUFFER, size(c, nv), c, GL_STATIC_DRAW);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(2);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffers[3]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, size(t, nt), t, GL_STATIC_DRAW);
    count = nt * 3;
  }

  ~GLMeshArray()
  {
    glDeleteBuffers(4, buffers);
    glDeleteVertexArrays(1, &vao);
  }

  void draw(const mat4f& m = mat4f::identity(), GLenum mode = GL_FILL)
  {
    auto mv = camera->getWorldToCameraMatrix() * m;
    const auto c = Color::blue;

    program->setUniform("mvMatrix", mv);
    program->setUniform("pMatrix", camera->getProjectionMatrix());
    program->setUniform("nMatrix", normalMatrix(m, camera));
    program->setUniform("useVertexColors", useVertexColors);
    program->setUniform("shadingMode", shadingMode);
    program->setUniform("material.Oa", 0.2f * c);
    program->setUniform("material.Od", 0.4f * c);
    program->setUniform("material.Os", 0.7f * c);
    program->setUniform("material.ns", float(1));
    program->setUniform("ambientLight", Color::white);
    program->setUniform("light.position", vec4(0.5f, 0.5f, 0, 1));
    program->setUniform("light.color", Color::white);
    glBindVertexArray(vao);
    glPolygonMode(GL_FRONT_AND_BACK, mode);
    glEnable(GL_DEPTH_TEST);
    glDrawElements(GL_TRIANGLES, count, GL_UNSIGNED_INT, 0);
    glDisable(GL_DEPTH_TEST);
  }

}; // GLMeshArray

bool
makeProgram(const char* name, const char* vs, const char* fs)
{
  if (program != nullptr && !strcmp(program->getName(), name))
    return false;
  delete program;
  printf("Making program '%s'\n", name);
  program = new GLSL::Program(name);
  program->addShader(GL_VERTEX_SHADER, GLSL::STRING, vs);
  program->addShader(GL_FRAGMENT_SHADER, GLSL::STRING, fs);
  program->use();
  return true;
}

inline void
clearScreen(const Color& c = Color::black)
{
  glClearColor(c.r, c.g, c.b, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void
setDrawPoint()
{
  static auto vs =
    "#version 330\n"
    "void main()\n"
    "{\n"
    "  gl_Position = vec4(0, 0, -2, 2);\n"
    "}";
  static auto fs =
    "#version 330\n"
    "out vec4 fragmentColor;\n"
    "void main()\n"
    "{\n"
    "  fragmentColor = vec4(0, 1, 0, 1);\n"
    "}";

  if (!makeProgram("One point", vs, fs))
    return;
  render = []()
  {
    clearScreen(Color::red);
    glPointSize(40);
    glDrawArrays(GL_POINTS, 0, 1);
  };
}

void
setDraw4Points()
{
  static auto vs =
    "#version 330\n"
    "uniform vec4 v[] = vec4[4]("
    "  vec4(-1, -1, 0, 1),"
    "  vec4(+1, -1, 0, 1),"
    "  vec4(+1, +1, 0, 1),"
    "  vec4(-1, +1, 0, 1));\n"
    "uniform vec4 c[] = vec4[4]("
    "  vec4(0, 1, 1, 1),"
    "  vec4(0, 1, 0, 1),"
    "  vec4(1, 1, 0, 1),"
    "  vec4(1, 0, 1, 1));\n"
    "out vec4 vertexColor;\n"
    "void main()\n"
    "{\n"
    "  gl_Position = v[gl_VertexID];\n"
    "  vertexColor = c[gl_VertexID];\n"
    "}";
  static auto fs =
    "#version 330\n"
    "in vec4 vertexColor;\n"
    "out vec4 fragmentColor;\n"
    "void main()\n"
    "{\n"
    "  fragmentColor = vertexColor;\n"
    "}";

  if (!makeProgram("Four points", vs, fs))
    return;
  render = []()
  {
    clearScreen(Color::red);
    glPointSize(40);
    glDrawArrays(GL_POINTS, 0, 4);
  };
}

void
setDrawLines()
{
  static auto vs =
    "#version 330\n"
    "uniform vec4 v[] = vec4[4]("
    "  vec4(-1, -1, 0, 1),"
    "  vec4(+1, +1, 0, 1),"
    "  vec4(+1, -1, 0, 1),"
    "  vec4(-1, +1, 0, 1));\n"
    "uniform vec4 c[] = vec4[4]("
    "  vec4(0, 1, 1, 1),"
    "  vec4(0, 1, 0, 1),"
    "  vec4(1, 1, 0, 1),"
    "  vec4(1, 0, 1, 1));\n"
    "out vec4 vertexColor;\n"
    "void main()\n"
    "{\n"
    "  gl_Position = v[gl_VertexID];\n"
    "  vertexColor = c[gl_VertexID];\n"
    "}";
  static auto fs =
    "#version 330\n"
    "in vec4 vertexColor;\n"
    "out vec4 fragmentColor;\n"
    "void main()\n"
    "{\n"
    "  fragmentColor = vertexColor;\n"
    "}";

  if (!makeProgram("Two lines", vs, fs))
    return;
  render = []()
  {
    clearScreen(Color::gray);
    glLineWidth(4);
    glDrawArrays(GL_LINES, 0, 4);
  };
}

void
setDrawTriangle()
{
  static auto vs =
    "#version 330\n"
    "uniform vec4 v[] = vec4[3]("
    "  vec4(-0.75, -0.75, -1, 1),"
    "  vec4(+0.75, -0.75, -1, 1),"
    "  vec4(0, +0.75, 0, 1));\n"
    "uniform vec4 c[] = vec4[3]("
    "  vec4(0, 1, 1, 1),"
    "  vec4(0, 1, 0, 1),"
    "  vec4(1, 0, 1, 1));\n"
    "out vec4 vertexColor;\n"
    "void main()\n"
    "{\n"
    "  gl_Position = v[gl_VertexID];\n"
    "  vertexColor = c[gl_VertexID];\n"
    "}";
  static auto fs =
    "#version 330\n"
    "in vec4 vertexColor;\n"
    "out vec4 fragmentColor;\n"
    "void main()\n"
    "{\n"
    "  fragmentColor = vertexColor;\n"
    "}";

  if (!makeProgram("Triangle", vs, fs))
    return;
  render = []()
  {
    clearScreen(Color::gray);
    glDrawArrays(GL_TRIANGLES, 0, 3);
  };
}

#define STRINGIFY(A) "#version 330\n"#A

void
setDrawBox()
{
  const vec4f p1(-0.5, -0.5, -0.5, 1);
  const vec4f p2(+0.5, -0.5, -0.5, 1);
  const vec4f p3(+0.5, +0.5, -0.5, 1);
  const vec4f p4(-0.5, +0.5, -0.5, 1);
  const vec4f p5(-0.5, -0.5, +0.5, 1);
  const vec4f p6(+0.5, -0.5, +0.5, 1);
  const vec4f p7(+0.5, +0.5, +0.5, 1);
  const vec4f p8(-0.5, +0.5, +0.5, 1);
  const vec3f n1(-1, 0, 0);
  const vec3f n2(+1, 0, 0);
  const vec3f n3(0, -1, 0);
  const vec3f n4(0, +1, 0);
  const vec3f n5(0, 0, -1);
  const vec3f n6(0, 0, +1);
  const Color c1(Color::black);
  const Color c2(Color::red);
  const Color c3(Color::yellow);
  const Color c4(Color::green);
  const Color c5(Color::blue);
  const Color c6(Color::magenta);
  const Color c7(Color::cyan);
  const Color c8(Color::white);

  static const vec4f v[]
  {
    p1, p5, p8, p4, // x = -0.5
    p2, p3, p7, p6, // x = +0.5
    p1, p2, p6, p5, // y = -0.5
    p4, p8, p7, p3, // y = +0.5
    p1, p4, p3, p2, // z = -0.5
    p5, p6, p7, p8  // z = +0.5
  };
  static const vec3f n[]
  {
    n1, n1, n1, n1, // x = -0.5
    n2, n2, n2, n2, // x = +0.5
    n3, n3, n3, n3, // y = -0.5
    n4, n4, n4, n4, // y = +0.5
    n5, n5, n5, n5, // z = -0.5
    n6, n6, n6, n6  // z = +0.5
  };
  static const Color c[]
  {
    c1, c5, c8, c4, // x = -0.5
    c2, c3, c7, c6, // x = +0.5
    c1, c2, c6, c5, // y = -0.5
    c4, c8, c7, c3, // y = +0.5
    c1, c4, c3, c2, // z = -0.5
    c5, c6, c7, c8  // z = +0.5
  };
  static const Triangle t[]
  {
    {  0,  1,  2 }, {  2,  3,  0 },
    {  4,  5,  7 }, {  5,  6,  7 },
    {  8,  9, 11 }, {  9, 10, 11 },
    { 12, 13, 14 }, { 14, 15, 12 },
    { 16, 17, 19 }, { 17, 18, 19 },
    { 20, 21, 22 }, { 22, 23, 20 }
  };

  static const char* vs = STRINGIFY(
    struct MaterialInfo
    {
      vec4 Oa;  // ambient color
      vec4 Od;  // diffuse color
      vec4 Os;  // specular spot color
      float ns; // specular shininess exponent
    };

    struct LightInfo
    {
      vec4 position; // light position in camera coordinates
      vec4 color;    // light color
    };

    layout(location = 0) in vec4 position;
    layout(location = 1) in vec3 normal;
    layout(location = 2) in vec4 color;
    uniform mat4 mvMatrix = mat4(1);
    uniform mat4 pMatrix = mat4(1);
    uniform mat3 nMatrix = mat3(1);
    uniform int useVertexColors;
    uniform int shadingMode;
    uniform MaterialInfo material;
    uniform LightInfo light;
    uniform vec4 ambientLight;
    out vec3 vertexPosition;
    out vec3 vertexNormal;
    out vec4 vertexColor;

    vec4 phong(vec3 P, vec3 N)
    {
      vec4 c = material.Oa * ambientLight;
      vec3 L = light.position.xyz;

      if (light.position.w != 0)
        L = normalize(P - L);

      vec3 R = reflect(-L, N);
      vec3 V = normalize(P);

      c += light.color * material.Od * max(dot(-L, N), 0);
      c += light.color * material.Os * pow(max(dot(-R, V), 0), material.ns);
      return c;
    }

    void main()
    {
      vec4 P = mvMatrix * position;

      gl_Position = pMatrix * P;
      if (useVertexColors != 0)
        vertexColor = color;
      else
      {
        vertexPosition = vec3(P);
        vertexNormal = normalize(nMatrix * normal);
        if (shadingMode == 0)
          vertexColor = phong(vertexPosition, vertexNormal);
      }
    }
  );
  static const char* fs = STRINGIFY(
    struct MaterialInfo
    {
      vec4 Oa;  // ambient color
      vec4 Od;  // diffuse color
      vec4 Os;  // specular spot color
      float ns; // specular shininess exponent
    };

    struct LightInfo
    {
      vec4 position; // light position in camera coordinates
      vec4 color;    // light color
    };

    uniform int useVertexColors;
    uniform int shadingMode;
    uniform MaterialInfo material;
    uniform LightInfo light;
    uniform vec4 ambientLight;
    in vec3 vertexPosition;
    in vec3 vertexNormal;
    in vec4 vertexColor;
    out vec4 fragmentColor;

    vec4 phong(vec3 P, vec3 N)
    {
      vec4 c = material.Oa * ambientLight;
      vec3 L = light.position.xyz;

      if (light.position.w != 0)
        L = normalize(P - L);

      vec3 R = reflect(-L, N);
      vec3 V = normalize(P);

      c += light.color * material.Od * max(dot(-L, N), 0);
      c += light.color * material.Os * pow(max(dot(-R, V), 0), material.ns);
      return c;
    }

    void main()
    {
      if (useVertexColors != 0 || shadingMode == 0)
        fragmentColor = vertexColor;
      else
        fragmentColor = phong(vertexPosition, vertexNormal);
    }
  );
  static GLMeshArray* mesh;

  if (!makeProgram("Box", vs, fs))
    return;
  if (mesh == nullptr)
    mesh = new GLMeshArray(v, n, c, 24, t, 12);
  render = []()
  {
    clearScreen(Color::gray);
    mesh->draw(mat4f::TRS(vec3f(+1, +1, 0), vec3f::null(), vec3f(2)));
    mesh->draw(mat4f::TRS(vec3f(-1, -1, 0), vec3f::null(), vec3f(5, 4, 1)));
  };
}

void
displayCallback()
{
  if (render != nullptr)
  {
    camera->updateView();
    render();
    glutSwapBuffers();
  }
}

void
reshapeCallback(int w, int h)
{
  camera->setAspectRatio(REAL(W = w) / REAL(H = h));
  glViewport(0, 0, W, H);
}

bool
processKeys()
{
  for (int i = 0; i < MAX_KEYS; i++)
  {
    if (!keys[i])
      continue;

    float len = 2 * camera->windowHeight() / float(H);

    switch (i)
    {
      // Camera controls
      case 'w':
        camera->move(0, 0, -len);
        break;
      case 's':
        camera->move(0, 0, +len);
        break;
      case 'q':
        camera->move(0, +len, 0);
        break;
      case 'z':
        camera->move(0, -len, 0);
        break;
      case 'a':
        camera->move(-len, 0, 0);
        break;
      case 'd':
        camera->move(+len, 0, 0);
        break;
      case '+':
        camera->zoom(ZOOM_SCALE);
        break;
      case '-':
        camera->zoom(1 / ZOOM_SCALE);
        break;
      case 'p':
        camera->changeProjectionType();
        break;
    }
  }
  return camera->isModified();
}

void
keyboardCallback(unsigned char key, int /*x*/, int /*y*/)
{
  keys[key] = true;
  if (processKeys())
    glutPostRedisplay();
}

void
keyboardUpCallback(unsigned char key, int /*x*/, int /*y*/)
{
  keys[key] = false;
  switch (key)
  {
    case 27:
      // glutLeaveMainLoop();
      // return;
      exit(EXIT_SUCCESS);
    case '1':
      setDrawPoint();
      break;
    case '2':
      setDraw4Points();
      break;
    case '3':
      setDrawLines();
      break;
    case '4':
      setDrawTriangle();
      break;
    case '5':
      setDrawBox();
      break;
    case 'c':
      useVertexColors ^= 1;
      break;
    case 'm':
      shadingMode ^= 1;
      break;
    default:
      return;
  }
  glutPostRedisplay();
}

void
mouseCallback(int, int, int x, int y)
{
  mouseX = x;
  mouseY = y;
}

void
motionCallback(int x, int y)
{
  const float ay = (mouseX - x) * CAMERA_RES / float(W) * !keys['x'];
  const float ax = (mouseY - y) * CAMERA_RES / float(H) * !keys['y'];

  mouseX = x;
  mouseY = y;
  if (ax != 0 || ay != 0)
  {
    keys['r'] ? camera->roll(ay) : camera->rotateYX(ay, ax);
    glutPostRedisplay();
  }
}

inline void
initGL(int argc, char** argv)
{
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
  glutInitWindowSize(WIN_W, WIN_H);
  glutCreateWindow("GL Test");
  GLSL::init();
  glutDisplayFunc(displayCallback);
  glutReshapeFunc(reshapeCallback);
  glutKeyboardFunc(keyboardCallback);
  glutKeyboardUpFunc(keyboardUpCallback);
  glutMouseFunc(mouseCallback);
  glutMotionFunc(motionCallback);
}

inline void
showOptions()
{
  printf(
    "Options\n"
    "-------\n"
    "(1) draw a single point\n"
    "(2) draw four points\n"
    "(3) draw two lines\n"
    "(4) draw a triangle\n"
    "(5) draw boxes with camera\n\n"
    "Camera controls\n"
    "---------------\n"
    "(w) pan forward  (s) pan backward\n"
    "(q) pan up       (z) pan down\n"
    "(a) pan left     (d) pan right\n"
    "(+) zoom in      (-) zoom out\n\n");
}

int
main(int argc, char** argv)
{
  initGL(argc, argv);
  camera = new Camera();
  showOptions();
  setDrawBox();
  glutMainLoop();
  /*
  puts("Press any key to exit...");
  getchar();
  */
  return 0;
}
