// FPS camera with Euler angles (yaw/pitch drive front/right/up); view via
// glm::lookAt, perspective projection with a Y-axis flip for Vulkan.
// Input: WASD + Space/Shift, mouse look, scroll wheel FOV (1-120 deg).

#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <GLFW/glfw3.h>

class Camera {
public:
    // position: camera start position (default (0, 0, 3)).
    // yaw: initial yaw; -90 faces -Z (yaw 0 faces +X).
    // pitch: initial pitch; 0 = level.
    Camera(glm::vec3 position = glm::vec3(0.0f, 0.0f, 3.0f),
           float yaw = -90.0f, float pitch = 0.0f);

    // View matrix via glm::lookAt(position, position + front, up).
    glm::mat4 getViewMatrix() const;

    // Perspective projection; aspect is width / height, nearPlane/farPlane are the
    // clip distances. proj[1][1] is negated to match Vulkan's flipped Y axis.
    glm::mat4 getProjectionMatrix(float aspect) const;

    // WASD movement + Space up + Shift down. deltaTime keeps speed frame-rate independent.
    void processKeyboard(GLFWwindow* window, float deltaTime);

    // Mouse look: updates yaw and pitch, then recomputes direction vectors.
    // xpos, ypos: current mouse position (pixels).
    void processMouse(double xpos, double ypos);

    // Scroll wheel adjusts FOV. yoffset: vertical scroll amount.
    void processScroll(double yoffset);

    // Resets mouse state (call when recapturing the mouse to avoid a view jump).
    // x, y: current mouse position.
    void resetMouse(double x, double y);

    // Getters / setters.
    glm::vec3 getPosition() const { return position; }
    float     getFov()      const { return fov; }
    void      setFov(float f) { fov = f; }
    float     getNearPlane() const { return nearPlane; }
    void      setNearPlane(float n) { nearPlane = n; }
    float     getFarPlane()  const { return farPlane; }
    void      setFarPlane(float f)  { farPlane = f; }

private:
    // Recomputes front, right, and up from yaw and pitch.
    void updateVectors();

    // Camera state.

    // Position and direction vectors.
    glm::vec3 position;  // World-space position.
    glm::vec3 front;     // Forward direction (from yaw/pitch).
    glm::vec3 up;        // Up direction (from front x worldUp).
    glm::vec3 right;     // Right direction (from front x worldUp).
    glm::vec3 worldUp;   // Fixed world up (0, 1, 0).

    // Euler angles (degrees).
    float yaw;    // Horizontal rotation around Y.
    float pitch;  // Vertical rotation around X.

    // Movement and view parameters.
    float speed       = 3.0f;   // Units per second.
    float sensitivity  = 0.1f;  // Pixels -> angle factor.
    float fov          = 45.0f; // Field of view (degrees, 1-120).
    float nearPlane    = 0.1f;  // Near clip plane.
    float farPlane     = 100.0f; // Far clip plane.

    // Mouse state.
    bool  firstMouse = true;           // True until the first mouse input (avoids an initial jump).
    double lastX = 0.0, lastY = 0.0;  // Previous mouse position (for offsets).
};
