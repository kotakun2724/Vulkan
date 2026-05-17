#version 450

// ============================================================================
//  Vulkan によるレイトレーシング (LT デモ)
//
//  本来の VK_KHR_ray_tracing_pipeline では下記のシェーダステージが
//  ハードウェアパイプラインとして用意される:
//
//      rayGenerationKHR  ─▶ traceRayEXT
//                            ├─ intersection / closestHit  (BLAS ごと)
//                            └─ miss
//
//  MoltenVK ではこれら拡張が動かないため、本ファイルでは
//  「同じ役割を 1 枚のフラグメントシェーダで手書きする」アプローチを採る。
//  各ブロックの見出しコメントは KHR RT のどのステージに対応するかを示す。
// ============================================================================

layout(location = 0) in  vec2 vUv;
layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform RTUbo {
    float time;
    float aspect;
    float cameraYaw;
    float _pad;
} u;

// ---------------------------------------------------------------------------
//  Scene (TLAS analogue)
// ---------------------------------------------------------------------------
struct Sphere {
    vec3  center;
    float radius;
    vec3  albedo;
    float reflectivity;   // 0 = pure diffuse, 1 = perfect mirror
};

const int NUM_SPHERES = 3;

Sphere getSphere(int i) {
    if (i == 0) return Sphere(vec3(-1.6, 0.55, 0.0), 0.55,
                              vec3(0.90, 0.30, 0.30), 0.05);   // red, diffuse
    if (i == 1) return Sphere(vec3( 0.0, 0.65, 0.0), 0.65,
                              vec3(0.85, 0.85, 0.95), 0.55);   // metal-ish
    return            Sphere(vec3( 1.6, 0.50, 0.2), 0.50,
                              vec3(0.95, 0.95, 0.95), 0.92);   // mirror
}

// Ground plane: y = 0
const vec3  PLANE_NORMAL = vec3(0.0, 1.0, 0.0);
const float PLANE_D      = 0.0;

const vec3  LIGHT_POS    = vec3(3.0, 4.5, 2.0);
const vec3  LIGHT_COLOR  = vec3(1.0);

const float TMIN = 1e-3;
const float TMAX = 1e4;

// ---------------------------------------------------------------------------
//  Intersection routines  (== rint shaders / built-in triangle test in KHR RT)
// ---------------------------------------------------------------------------
float intersectSphere(vec3 ro, vec3 rd, Sphere s) {
    vec3 oc = ro - s.center;
    float b = dot(oc, rd);
    float c = dot(oc, oc) - s.radius * s.radius;
    float h = b * b - c;
    if (h < 0.0) return -1.0;
    h = sqrt(h);
    float t = -b - h;
    if (t > TMIN) return t;
    t = -b + h;
    return t > TMIN ? t : -1.0;
}

float intersectPlane(vec3 ro, vec3 rd) {
    float denom = dot(rd, PLANE_NORMAL);
    if (abs(denom) < 1e-4) return -1.0;
    float t = -(dot(ro, PLANE_NORMAL) + PLANE_D) / denom;
    return t > TMIN ? t : -1.0;
}

// ---------------------------------------------------------------------------
//  traceRay  ==  traceRayEXT (manual TLAS traversal)
// ---------------------------------------------------------------------------
struct Hit {
    float t;
    vec3  position;
    vec3  normal;
    vec3  albedo;
    float reflectivity;
    bool  isPlane;
};

bool traceRay(vec3 ro, vec3 rd, out Hit hit) {
    hit.t            = TMAX;
    hit.position     = vec3(0.0);
    hit.normal       = vec3(0.0, 1.0, 0.0);
    hit.albedo       = vec3(0.0);
    hit.reflectivity = 0.0;
    hit.isPlane      = false;
    bool found = false;

    for (int i = 0; i < NUM_SPHERES; ++i) {
        Sphere s = getSphere(i);
        float t = intersectSphere(ro, rd, s);
        if (t > 0.0 && t < hit.t) {
            hit.t            = t;
            hit.position     = ro + rd * t;
            hit.normal       = normalize(hit.position - s.center);
            hit.albedo       = s.albedo;
            hit.reflectivity = s.reflectivity;
            hit.isPlane      = false;
            found = true;
        }
    }

    float tp = intersectPlane(ro, rd);
    if (tp > 0.0 && tp < hit.t) {
        hit.t        = tp;
        hit.position = ro + rd * tp;
        hit.normal   = PLANE_NORMAL;
        // Procedural checker pattern for the ground.
        vec2 p = hit.position.xz;
        float check = mod(floor(p.x) + floor(p.y), 2.0);
        hit.albedo       = mix(vec3(0.22), vec3(0.75), check);
        hit.reflectivity = 0.08;
        hit.isPlane      = true;
        found = true;
    }

    return found;
}

// Shadow ray: only needs an "any-hit" style early-out.
bool traceShadow(vec3 ro, vec3 rd, float maxDist) {
    for (int i = 0; i < NUM_SPHERES; ++i) {
        Sphere s = getSphere(i);
        float t = intersectSphere(ro, rd, s);
        if (t > 0.0 && t < maxDist) return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
//  Miss shader analogue: simple sky gradient
// ---------------------------------------------------------------------------
vec3 missShade(vec3 rd) {
    float t = 0.5 * (rd.y + 1.0);
    return mix(vec3(1.0, 1.0, 1.0), vec3(0.45, 0.70, 1.00), t);
}

// ---------------------------------------------------------------------------
//  Closest-hit shader analogue: Lambert + Blinn-Phong + shadow ray
// ---------------------------------------------------------------------------
vec3 closestHitShade(Hit hit, vec3 rd) {
    vec3 toLight   = LIGHT_POS - hit.position;
    float distLight = length(toLight);
    vec3 L = toLight / distLight;

    float shadow = traceShadow(hit.position + hit.normal * 1e-3, L, distLight)
                       ? 0.18 : 1.0;

    float ndotl   = max(dot(hit.normal, L), 0.0);
    vec3  diffuse = hit.albedo * ndotl;

    vec3 V = -rd;
    vec3 H = normalize(L + V);
    float spec = pow(max(dot(hit.normal, H), 0.0), 64.0);

    vec3 ambient = hit.albedo * 0.08;
    return ambient + (diffuse + vec3(spec) * 0.4) * LIGHT_COLOR * shadow;
}

// ---------------------------------------------------------------------------
//  rayGen shader analogue: primary ray + 1 reflection bounce
// ---------------------------------------------------------------------------
void main() {
    // [1] Screen-space -> NDC (Vulkan: top-left origin, y-flip)
    vec2 uv = vUv * 2.0 - 1.0;
    uv.x *= u.aspect;
    uv.y = -uv.y;

    // [2] Orbit camera around the origin driven by cameraYaw.
    float yaw      = u.cameraYaw;
    vec3  ro       = vec3(sin(yaw) * 4.5, 1.7, cos(yaw) * 4.5);
    vec3  target   = vec3(0.0, 0.55, 0.0);
    vec3  forward  = normalize(target - ro);
    vec3  right    = normalize(cross(forward, vec3(0.0, 1.0, 0.0)));
    vec3  up       = cross(right, forward);

    float fov   = radians(55.0);
    float halfT = tan(fov * 0.5);
    vec3  rd    = normalize(forward + right * uv.x * halfT + up * uv.y * halfT);

    // [3] Iterative reflection (replaces recursive traceRayEXT calls)
    vec3 color      = vec3(0.0);
    vec3 throughput = vec3(1.0);

    const int MAX_BOUNCES = 3;
    for (int bounce = 0; bounce < MAX_BOUNCES; ++bounce) {
        Hit hit;
        if (traceRay(ro, rd, hit)) {
            vec3 shaded = closestHitShade(hit, rd);
            color += throughput * shaded * (1.0 - hit.reflectivity);

            if (hit.reflectivity <= 0.0) break;
            throughput *= hit.reflectivity * mix(vec3(1.0), hit.albedo, 0.5);
            ro = hit.position + hit.normal * 1e-3;
            rd = reflect(rd, hit.normal);
        } else {
            color += throughput * missShade(rd);
            break;
        }
    }

    // [4] Gamma correct to sRGB-ish output.
    color = pow(color, vec3(1.0 / 2.2));
    outColor = vec4(color, 1.0);
}
