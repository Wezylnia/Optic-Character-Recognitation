"""
Metin kutusu siralama ve satir gruplama yardimci fonksiyonlari
"""

import cv2
import math
import numpy as np
from typing import List, Tuple


def sort_boxes_by_position(
    boxes: List[np.ndarray],
    line_threshold: int = 10
) -> List[np.ndarray]:
    """
    Kutulari okuma sirasina gore sirala (sol-ustten sag-alta)

    Args:
        boxes: Kutu listesi
        line_threshold: Ayni satir icin Y toleransi

    Returns:
        Siralanmis kutu listesi
    """
    if not boxes:
        return boxes

    boxes_with_pos = []
    for box in boxes:
        center_y = np.mean(box[:, 1])
        center_x = np.mean(box[:, 0])
        boxes_with_pos.append((box, center_x, center_y))

    boxes_with_pos.sort(key=lambda x: x[2])

    lines = []
    current_line = [boxes_with_pos[0]]
    current_y = boxes_with_pos[0][2]

    for item in boxes_with_pos[1:]:
        if abs(item[2] - current_y) < line_threshold:
            current_line.append(item)
        else:
            lines.append(current_line)
            current_line = [item]
            current_y = item[2]

    lines.append(current_line)

    sorted_boxes = []
    for line in lines:
        line.sort(key=lambda x: x[1])
        sorted_boxes.extend([item[0] for item in line])

    return sorted_boxes


def get_box_rotation_angle(box: np.ndarray) -> float:
    """
    Kutunun rotasyon acisini hesapla

    Args:
        box: 4 noktali polygon [4, 2]

    Returns:
        Rotasyon acisi (derece, -45 ile 45 arasi)
    """
    rect = cv2.minAreaRect(box.astype(np.float32))
    angle = rect[2]

    width, height = rect[1]
    if width < height:
        angle = angle - 90

    while angle < -45:
        angle += 90
    while angle > 45:
        angle -= 90

    return angle


def order_points(pts: np.ndarray) -> np.ndarray:
    """Noktalari saat yonunde sirala (sol-ust'ten baslayarak)"""
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # sol-ust
    rect[2] = pts[np.argmax(s)]   # sag-alt

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # sag-ust
    rect[3] = pts[np.argmax(diff)]  # sol-alt

    return rect


def crop_polygon(image: np.ndarray, box: np.ndarray) -> np.ndarray:
    """
    Polygon seklindeki bolgeden crop al (perspektif duzeltmeli)

    Args:
        image: Kaynak gorsel
        box: 4 noktali polygon [4, 2]

    Returns:
        Crop edilmis gorsel
    """
    pts = order_points(box)

    width = int(max(
        np.linalg.norm(pts[0] - pts[1]),
        np.linalg.norm(pts[2] - pts[3])
    ))
    height = int(max(
        np.linalg.norm(pts[0] - pts[3]),
        np.linalg.norm(pts[1] - pts[2])
    ))

    if width <= 0 or height <= 0:
        return np.zeros((32, 100, 3), dtype=np.uint8)

    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(pts.astype(np.float32), dst)
    warped = cv2.warpPerspective(image, matrix, (width, height))

    return warped


def correct_box_rotation(
    image: np.ndarray,
    box: np.ndarray,
    angle_threshold: float = 5.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Text region icin rotasyon duzeltmesi

    Args:
        image: Kaynak gorsel
        box: Text bounding box [4, 2]
        angle_threshold: Duzeltme yapilacak minimum aci

    Returns:
        (duzeltilmis_crop, yeni_box)
    """
    angle = get_box_rotation_angle(box)

    if abs(angle) < angle_threshold:
        return crop_polygon(image, box), box

    x_min = int(max(0, np.min(box[:, 0]) - 5))
    y_min = int(max(0, np.min(box[:, 1]) - 5))
    x_max = int(min(image.shape[1], np.max(box[:, 0]) + 5))
    y_max = int(min(image.shape[0], np.max(box[:, 1]) + 5))

    region = image[y_min:y_max, x_min:x_max]
    if region.size == 0:
        return crop_polygon(image, box), box

    local_box = box.copy()
    local_box[:, 0] -= x_min
    local_box[:, 1] -= y_min

    h, w = region.shape[:2]
    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = np.abs(matrix[0, 0])
    sin = np.abs(matrix[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    matrix[0, 2] += (new_w - w) / 2
    matrix[1, 2] += (new_h - h) / 2

    rotated = cv2.warpAffine(region, matrix, (new_w, new_h), borderValue=(255, 255, 255))

    ones = np.ones((4, 1))
    points = np.hstack([local_box, ones])
    new_box = (matrix @ points.T).T.astype(np.float32)

    x1 = max(0, int(np.min(new_box[:, 0])))
    y1 = max(0, int(np.min(new_box[:, 1])))
    x2 = min(new_w, int(np.max(new_box[:, 0])))
    y2 = min(new_h, int(np.max(new_box[:, 1])))

    cropped = rotated[y1:y2, x1:x2]
    if cropped.size == 0:
        return crop_polygon(image, box), box

    return cropped, new_box


class AdaptiveLineGrouper:
    """
    Adaptif satir gruplama

    Sabit threshold yerine box yuksekliklerine gore dinamik gruplama yapar.
    """

    def __init__(
        self,
        overlap_threshold: float = 0.5,
        y_tolerance_ratio: float = 0.5
    ):
        """
        Args:
            overlap_threshold: Ayni satir icin minimum Y overlap orani
            y_tolerance_ratio: Box yuksekligine gore Y toleransi orani
        """
        self.overlap_threshold = overlap_threshold
        self.y_tolerance_ratio = y_tolerance_ratio

    def group_into_lines(
        self,
        boxes: List[np.ndarray]
    ) -> List[List[np.ndarray]]:
        """
        Kutulari satirlara grupla

        Args:
            boxes: Kutu listesi

        Returns:
            Satir listesi, her satir kutu listesi icerir
        """
        if not boxes:
            return []

        box_info = []
        for box in boxes:
            y_min = np.min(box[:, 1])
            y_max = np.max(box[:, 1])
            x_center = np.mean(box[:, 0])
            height = y_max - y_min
            box_info.append({
                'box': box,
                'y_min': y_min,
                'y_max': y_max,
                'y_center': (y_min + y_max) / 2,
                'x_center': x_center,
                'height': height
            })

        box_info.sort(key=lambda x: x['y_center'])

        lines = []
        used = set()

        for i, info in enumerate(box_info):
            if i in used:
                continue

            current_line = [info]
            used.add(i)

            for j, other in enumerate(box_info):
                if j in used:
                    continue
                if self._is_same_line(info, other):
                    current_line.append(other)
                    used.add(j)

            lines.append(current_line)

        sorted_lines = []
        for line in lines:
            line.sort(key=lambda x: x['x_center'])
            sorted_lines.append([item['box'] for item in line])

        sorted_lines.sort(key=lambda line: np.mean([np.mean(box[:, 1]) for box in line]))

        return sorted_lines

    def _is_same_line(self, box1: dict, box2: dict) -> bool:
        """Iki kutunun ayni satirda olup olmadigini kontrol et"""
        overlap_start = max(box1['y_min'], box2['y_min'])
        overlap_end = min(box1['y_max'], box2['y_max'])
        overlap = max(0, overlap_end - overlap_start)

        height1 = box1['height']
        height2 = box2['height']
        min_height = min(height1, height2)

        if min_height <= 0:
            return False

        overlap_ratio = overlap / min_height

        avg_height = (height1 + height2) / 2
        y_diff = abs(box1['y_center'] - box2['y_center'])
        y_tolerance = avg_height * self.y_tolerance_ratio

        return overlap_ratio >= self.overlap_threshold or y_diff <= y_tolerance

    def group_and_sort(
        self,
        boxes: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Kutulari satirlara grupla ve duz liste olarak dondur

        Args:
            boxes: Kutu listesi

        Returns:
            Okuma sirasina gore siralanmis kutu listesi
        """
        lines = self.group_into_lines(boxes)
        result = []
        for line in lines:
            result.extend(line)
        return result


def adaptive_sort_boxes(boxes: List[np.ndarray]) -> List[np.ndarray]:
    """
    Kutulari adaptif olarak okuma sirasina gore sirala

    Args:
        boxes: Kutu listesi

    Returns:
        Siralanmis kutu listesi
    """
    grouper = AdaptiveLineGrouper()
    return grouper.group_and_sort(boxes)


def group_boxes_into_lines(boxes: List[np.ndarray]) -> List[List[np.ndarray]]:
    """
    Kutulari satirlara grupla

    Args:
        boxes: Kutu listesi

    Returns:
        Satir listesi
    """
    grouper = AdaptiveLineGrouper()
    return grouper.group_into_lines(boxes)
