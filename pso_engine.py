import cv2
import numpy as np
from sklearn.cluster import KMeans
import time


def load_image(image_path, color=True):
    if color:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
    return image


def _build_feature_vectors(image, spatial_weight=0.3):
    h, w = image.shape[:2]

    yy, xx = np.mgrid[0:h, 0:w]
    y_norm = (yy / h * 255 * spatial_weight).astype(np.float64)
    x_norm = (xx / w * 255 * spatial_weight).astype(np.float64)

    if len(image.shape) == 3:
        pixels = image.reshape(-1, 3).astype(np.float64)
        spatial = np.column_stack([x_norm.ravel(), y_norm.ravel()])
        features = np.hstack([pixels, spatial])
    else:
        pixels = image.reshape(-1, 1).astype(np.float64)
        spatial = np.column_stack([x_norm.ravel(), y_norm.ravel()])
        features = np.hstack([pixels, spatial])

    return features


def _resize_for_processing(image, max_pixels=50000):
    h, w = image.shape[:2]
    total = h * w
    if total > max_pixels:
        scale = np.sqrt(max_pixels / total)
        new_w = max(int(w * scale), 1)
        new_h = max(int(h * scale), 1)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image


def _sample_pixels(features, max_samples=8000):
    if len(features) > max_samples:
        indices = np.random.choice(len(features), max_samples, replace=False)
        return features[indices]
    return features


class PSOSegmenter:

    def __init__(self, num_centroids, num_channels, swarm_size=20,
                 max_iter=40, w=0.5, c1=1.5, c2=1.5):
        self.num_centroids = num_centroids
        self.num_channels = num_channels
        self.dim = num_centroids * num_channels
        self.swarm_size = swarm_size
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def _init_swarm_smart(self, sample_pixels):
        positions = np.zeros((self.swarm_size, self.dim))

        try:
            kmeans = KMeans(n_clusters=self.num_centroids, init='k-means++',
                            n_init=1, max_iter=10)
            kmeans.fit(sample_pixels)
            positions[0] = kmeans.cluster_centers_.ravel()
        except Exception:
            positions[0] = np.random.uniform(0, 255, self.dim)

        for i in range(1, self.swarm_size):
            noise = np.random.normal(0, 25, self.dim)
            positions[i] = np.clip(positions[0] + noise, 0, 255)

        return positions

    def _compute_fitness(self, position, pixels):
        centroids = position.reshape(self.num_centroids, self.num_channels)
        dists = np.linalg.norm(
            pixels[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2
        )
        labels = np.argmin(dists, axis=1)
        mse = np.mean(np.min(dists, axis=1) ** 2)
        return mse

    def optimize(self, pixels, progress_callback=None):
        sample = _sample_pixels(pixels, max_samples=5000)

        positions = self._init_swarm_smart(sample)
        velocities = np.random.uniform(-10, 10, (self.swarm_size, self.dim))

        fitness = np.array([self._compute_fitness(p, sample) for p in positions])
        pbest_pos = positions.copy()
        pbest_fit = fitness.copy()

        gbest_idx = np.argmin(fitness)
        gbest_pos = positions[gbest_idx].copy()
        gbest_fit = fitness[gbest_idx]

        no_improve_count = 0
        prev_best = gbest_fit

        for iteration in range(self.max_iter):
            r1 = np.random.rand(self.swarm_size, self.dim)
            r2 = np.random.rand(self.swarm_size, self.dim)

            w_decay = self.w * (1 - iteration / self.max_iter * 0.5)

            velocities = (w_decay * velocities +
                          self.c1 * r1 * (pbest_pos - positions) +
                          self.c2 * r2 * (gbest_pos - positions))

            velocities = np.clip(velocities, -30, 30)
            positions = np.clip(positions + velocities, 0, 255)

            fitness = np.array([self._compute_fitness(p, sample) for p in positions])

            improved = fitness < pbest_fit
            pbest_pos[improved] = positions[improved]
            pbest_fit[improved] = fitness[improved]

            current_best_idx = np.argmin(pbest_fit)
            if pbest_fit[current_best_idx] < gbest_fit:
                gbest_pos = pbest_pos[current_best_idx].copy()
                gbest_fit = pbest_fit[current_best_idx]
                no_improve_count = 0
            else:
                no_improve_count += 1

            if progress_callback:
                pct = min((iteration + 1) / self.max_iter * 100, 95)
                progress_callback(iteration + 1, self.max_iter, gbest_fit, pct)

            if no_improve_count > 8:
                break

        return gbest_pos.reshape(self.num_centroids, self.num_channels)


def segment_image(image_path, num_centroids, num_particles=None,
                  color=True, max_iter=40, progress_callback=None):
    if num_particles is None:
        num_particles = max(num_centroids * 4, 15)

    start_time = time.time()

    image = load_image(image_path, color)
    original_shape = image.shape

    proc_image = _resize_for_processing(image, max_pixels=40000)

    features = _build_feature_vectors(proc_image, spatial_weight=0.35)
    num_channels = features.shape[1]

    pso = PSOSegmenter(
        num_centroids=num_centroids,
        num_channels=num_channels,
        swarm_size=num_particles,
        max_iter=max_iter
    )
    centroids = pso.optimize(features, progress_callback=progress_callback)

    full_features = _build_feature_vectors(image, spatial_weight=0.35)

    dists = np.linalg.norm(
        full_features[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2
    )
    labels = np.argmin(dists, axis=1)

    if len(original_shape) == 3:
        color_centroids = centroids[:, :3]
        segmented = color_centroids[labels].reshape(original_shape)
        segmented = cv2.cvtColor(segmented.astype(np.uint8), cv2.COLOR_RGB2BGR)
    else:
        color_centroids = centroids[:, :1]
        segmented = color_centroids[labels].reshape(original_shape)
        segmented = segmented.astype(np.uint8)

    processing_time = round(time.time() - start_time, 2)

    if progress_callback:
        progress_callback(max_iter, max_iter, 0, 100)

    return {
        "segmented_image": segmented,
        "processing_time": processing_time,
        "dimensions": f"{original_shape[1]}x{original_shape[0]}",
        "num_centroids": num_centroids,
    }
