# Vulkan Triangle (macOS + MoltenVK)

macOSでVulkan SDK（MoltenVK同梱）を使い、GLFWでウィンドウを作成して三角形を描画します。

## スクリーンショット

![Vulkan Triangle](<スクリーンショット 2026-01-17 16.14.40.png>)

## 事前準備

### 1. Vulkan SDK のインストール
1. LunarGのVulkan SDK（macOS版）をインストール  
   https://vulkan.lunarg.com/sdk/home
2. インストール後に環境変数を設定（zshの例）

```
export VULKAN_SDK=/path/to/VulkanSDK/<version>/macOS
export PATH="$VULKAN_SDK/bin:$PATH"
export DYLD_LIBRARY_PATH="$VULKAN_SDK/lib:$DYLD_LIBRARY_PATH"
```

### 2. 依存ライブラリ（GLFW）
Homebrewを使う場合：

```
brew install glfw
```

## ビルド手順

```
mkdir -p build
cd build
cmake ..
cmake --build .
```

## 実行

```
./vulkan_triangle
```

## 補足
- シェーダは `shaders/` にあり、ビルド時に `glslc` でSPIR-Vへ変換されます。
- 画面をリサイズするとスワップチェーンを再作成して描画が継続します。

