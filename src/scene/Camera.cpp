// FPS camera implementation: constructor and direction init, view/projection
// matrices, keyboard/mouse/scroll input, and yaw/pitch -> direction updateVectors.

#include "Camera.h"
#include <cmath> // std::cos / std::sin

// Sets position, Euler angles, and world up, then computes the initial
// front/right/up from yaw/pitch via updateVectors().
Camera::Camera(glm::vec3 position, float yaw, float pitch)
    : position(position)
      ,
      yaw(yaw) // Default -90 deg, facing -Z.
      ,
      pitch(pitch) // Default 0 deg (level).
      ,
      worldUp(0.0f, 1.0f, 0.0f) // World up fixed to +Y.
{
    updateVectors();
}

// World -> camera space via glm::lookAt.
glm::mat4 Camera::getViewMatrix() const
{
    return glm::lookAt(position, position + front, up);
}

// Perspective projection; distant objects appear smaller.
glm::mat4 Camera::getProjectionMatrix(float aspect) const
{
    glm::mat4 proj = glm::perspective(glm::radians(fov), aspect, nearPlane, farPlane);

    // Vulkan clip space Y points down, opposite to OpenGL, so flip proj[1][1].
    proj[1][1] *= -1.0f;
    return proj;
}

// WASD + Space/Shift movement. velocity = speed x deltaTime keeps motion
// frame-rate independent.
void Camera::processKeyboard(GLFWwindow *window, float deltaTime)
{
    float velocity = speed * deltaTime;

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        position += front * velocity;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        position -= front * velocity;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        position -= right * velocity;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        position += right * velocity;
    // Rise along world up (not camera up) so elevation stays on world Y.
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
        position += worldUp * velocity;
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
        position -= worldUp * velocity;
}

// Mouse motion -> angle change: yaw += xoffset * sensitivity,
// pitch += yoffset * sensitivity. Pitch is clamped to +/-89 deg to avoid gimbal lock.
void Camera::processMouse(double xpos, double ypos)
{
    // First input only records the position, avoiding a view jump on capture.
    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
        return;
    }

    // xoffset: horizontal delta. yoffset uses lastY - ypos because screen Y
    // points down while camera pitch is positive upward.
    float xoffset = static_cast<float>(xpos - lastX) * sensitivity;
    float yoffset = static_cast<float>(lastY - ypos) * sensitivity;
    lastX = xpos;
    lastY = ypos;

    yaw += xoffset;
    pitch += yoffset;

    // Clamp pitch to +/-89 deg: at +/-90 the front vector becomes parallel to
    // worldUp and the cross products degenerate.
    if (pitch > 89.0f)
        pitch = 89.0f;
    if (pitch < -89.0f)
        pitch = -89.0f;

    updateVectors();
}

// Scroll up -> FOV shrinks (zoom in); scroll down -> FOV grows (zoom out).
void Camera::processScroll(double yoffset)
{
    fov -= static_cast<float>(yoffset);
    // Clamp FOV to [1, 120] degrees.
    if (fov < 1.0f)
        fov = 1.0f;
    if (fov > 120.0f)
        fov = 120.0f;
}

// Call when the mouse is recaptured: the next processMouse only records the
// position, so the view doesn't jump.
void Camera::resetMouse(double x, double y)
{
    lastX = x;
    lastY = y;
    firstMouse = true;
}

// Spherical -> Cartesian conversion: yaw/pitch to the front direction.
// front.x = cos(yaw) * cos(pitch)
// front.y = sin(pitch)
// front.z = sin(yaw) * cos(pitch)
void Camera::updateVectors()
{
    // Precompute trig: radians converted once, cos(pitch) computed once.
    // Equivalent to per-component calls, but explicit and also effective in debug builds.
    float yawR = glm::radians(yaw);
    float pitchR = glm::radians(pitch);
    float cy = std::cos(yawR);
    float sy = std::sin(yawR);
    float cp = std::cos(pitchR);
    float sp = std::sin(pitchR);

    glm::vec3 newFront;
    newFront.x = cy * cp;
    newFront.y = sp;
    newFront.z = sy * cp;
    front = glm::normalize(newFront);

    right = glm::normalize(glm::cross(front, worldUp));

    up = glm::normalize(glm::cross(right, front));
}
