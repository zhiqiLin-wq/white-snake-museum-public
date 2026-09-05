// ==========  (WGS84 -> GCJ-02) ==========
const A = 6378245.0
const EE = 0.00669342162296594323

function transformLat(x: number, y: number): number {
  let ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * Math.sqrt(Math.abs(x))
  ret += (20.0 * Math.sin(6.0 * x * Math.PI) + 20.0 * Math.sin(2.0 * x * Math.PI)) * 2.0 / 3.0
  ret += (20.0 * Math.sin(y * Math.PI) + 40.0 * Math.sin(y / 3.0 * Math.PI)) * 2.0 / 3.0
  ret += (160.0 * Math.sin(y / 12.0 * Math.PI) + 320 * Math.sin(y * Math.PI / 30.0)) * 2.0 / 3.0
  return ret
}

function transformLng(x: number, y: number): number {
  let ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * Math.sqrt(Math.abs(x))
  ret += (20.0 * Math.sin(6.0 * x * Math.PI) + 20.0 * Math.sin(2.0 * x * Math.PI)) * 2.0 / 3.0
  ret += (20.0 * Math.sin(x * Math.PI) + 40.0 * Math.sin(x / 3.0 * Math.PI)) * 2.0 / 3.0
  ret += (150.0 * Math.sin(x / 12.0 * Math.PI) + 300.0 * Math.sin(x / 30.0 * Math.PI)) * 2.0 / 3.0
  return ret
}

export function wgs84ToGcj02(lng: number, lat: number): [number, number] {
  const dLat = transformLat(lng - 105.0, lat - 35.0)
  const dLng = transformLng(lng - 105.0, lat - 35.0)
  const radLat = lat / 180.0 * Math.PI
  const magic = Math.sin(radLat)
  const magic2 = 1 - EE * magic * magic
  const sqrtMagic = Math.sqrt(magic2)
  const dLatFinal = (dLat * 180.0) / ((A * (1 - EE)) / (magic2 * sqrtMagic) * Math.PI)
  const dLngFinal = (dLng * 180.0) / (A / sqrtMagic * Math.cos(radLat) * Math.PI)
  return [lng + dLngFinal, lat + dLatFinal]
}

// ==========  WGS84  ==========
const rawCoords: Record<string, [number, number]> = {
  '峨眉山': [29.564, 103.388],
  '青城山': [30.918, 103.607],
  '西湖':   [30.2741, 120.1551],
  '金山寺': [32.2106, 119.4485],
  '断桥': [30.2409, 120.1512],
  '雷峰塔':   [30.2608, 120.1442],
  '卧佛寺': [38.467, 106.273],
  '承天寺': [25.112, 99.169],
  '望江楼': [30.628, 104.089],
  '龙虎山': [28.108, 117.016],
  '灵隐寺': [30.2408, 120.0957],
  '钱塘门': [30.26, 120.16],
}

// ==========  GCJ-02  (lat, lng  Leaflet) ==========
export const coords: Record<string, [number, number]> = {}
for (const [name, [lat, lng]] of Object.entries(rawCoords)) {
  const [gcjLng, gcjLat] = wgs84ToGcj02(lng, lat)
  coords[name] = [gcjLat, gcjLng]
}

// ========== 720 URL ==========
export const panoramaUrls: Record<string, string> = {
  '峨眉山': 'https://www.720yun.com/t/24vktepm5fw?scene_id=69856263',
  '青城山': 'https://www.720yun.com/t/2cdjvptntn1?scene_id=19436745',
  '西湖':   'https://www.720yun.com/t/51vkOlqldfq?scene_id=51248758',
  '金山寺': 'https://www.720yun.com/t/f2vkbh79g7y?scene_id=95860074',
  '断桥': 'https://www.720yun.com/t/b0vkObdmgpy?scene_id=58663613',
  '雷峰塔': 'https://www.720yun.com/t/6a8jOrwkuv3?scene_id=2082123',
  '卧佛寺': 'https://www.720yun.com/t/78akn7dq58q?scene_id=116153811',
  '承天寺': 'https://www.720yun.com/t/9dcjvOmf5a9?scene_id=20690149',
  '望江楼': 'https://www.720yun.com/t/0e4jt5unzO5?scene_id=13200053',
  '龙虎山': 'https://www.720yun.com/t/0232fjO8wea?scene_id=422669',
  '灵隐寺': 'https://www.720yun.com/t/de328ur6cts?scene_id=1773977',
  '钱塘门':   'https://www.720yun.com/t/96vk6li9r87?scene_id=98633220',
}

// ==========  ==========
export const storySegments: [string, string][] = [
  ['西湖', '白蛇传'],
  ['雷峰塔', '白蛇传'],
  ['金山寺', '白蛇传'],
  ['断桥', '白蛇传'],
  ['灵隐寺', '白蛇传'],
  ['峨眉山', '白蛇传'],
  ['青城山', '白蛇传'],
  ['望江楼', '白蛇传'],
  ['龙虎山', '白蛇传'],
  ['承天寺', '白蛇传'],
  ['卧佛寺', '白蛇传'],
]

// ==========  ==========
export function getBrowserZoomLevel(): number {
  const dpr = window.devicePixelRatio || 1
  const mqString = `(resolution: ${dpr}dppx)`
  if (window.matchMedia(mqString).matches) return 1
  const screenWidth = window.screen.width
  const innerW = window.innerWidth
  if (screenWidth && innerW) {
    const estimated = screenWidth / innerW
    if (estimated > 0.8 && estimated < 2) return Math.round(estimated * 10) / 10
  }
  return 1
}

export function computeBaseFontSize(): number {
  const zoom = getBrowserZoomLevel()
  const rawSize = 16 / (zoom * zoom)
  return Math.min(80, Math.max(12, rawSize))
}
