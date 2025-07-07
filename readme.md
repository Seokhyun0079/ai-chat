# AI Chat アプリケーション

パーソナライズされた AI チャットボットのためのフルスタック Web アプリケーションです。SKT KoGPT2 モデルをベースにした日本語対話型 AI システムです。

## 🚀 プロジェクト概要

このプロジェクトは、カカオトークの対話記録を学習データとして使用し、ユーザーの話し方や性格を模倣するパーソナライズされた AI チャットボットを作成することを目的としています。FastAPI バックエンドと Next.js フロントエンドで構成された現代的な Web アプリケーションです。

### 📹 デモ動画

プロジェクトの動作を確認するには、以下のデモ動画をご覧ください：

![AI Chat Demo](sample.mp4)

_デ모動画: AI Chat アプリケーションの基本的な機能と使用方法_

## 📁 プロジェクト構造

```
ai-chat/
├── api/                    # FastAPIバックエンド
│   ├── api.py             # メインAPIサーバー
│   ├── model_selector.py  # AIモデル選択・設定
│   ├── trainer.py         # モデル訓練ロジック
│   ├── requirements.txt   # Python依存関係
│   └── Dockerfile         # バックエンドコンテナ設定
├── front/                 # Next.jsフロントエンド
│   ├── app/              # Next.jsアプリディレクトリ
│   ├── package.json      # Node.js依存関係
│   └── Dockerfile        # フロントエンドコンテナ設定
├── docker-compose.yml    # 全体システムオーケストレーション
└── results/              # 訓練済みモデルチェックポイント
```

## 🛠️ 技術スタック

### バックエンド

- **FastAPI**: 高性能 Python Web フレームワーク
- **PyTorch**: ディープラーニングフレームワーク
- **Transformers**: Hugging Face トランスフォーマーライブラリ
- **SKT KoGPT2**: 韓国語 GPT2 モデル

### フロントエンド

- **Next.js 15**: React ベースのフルスタックフレームワーク
- **TypeScript**: 型安全性
- **Tailwind CSS**: ユーティリティファースト CSS フレームワーク

### インフラ

- **Docker**: コンテナ化
- **Docker Compose**: マルチコンテナオーケストレーション

## 🚀 はじめに

### 前提条件

- Docker および Docker Compose
- Node.js 18+ (ローカル開発用)
- Python 3.8+ (ローカル開発用)

### Docker を使用した実行 (推奨)

1. リポジトリのクローン

```bash
git clone <repository-url>
cd ai-chat
```

2. Docker Compose で実行

```bash
docker-compose up --build
```

3. ブラウザでアクセス

- フロントエンド: http://localhost:3000
- バックエンド API: http://localhost:8000

### ローカル開発環境

#### バックエンド設定

```bash
cd api
pip install -r requirements.txt
python api.py
```

#### フロントエンド設定

```bash
cd front
npm install
npm run dev
```

## 📡 API エンドポイント

### POST /chat/

チャットボットとの対話のためのメインエンドポイント

**リクエストボディ:**

```json
{
  "prompt": "ユーザーメッセージ",
  "previous_message": "前の会話内容",
  "max_length": 300
}
```

**レスポンス:**

```json
{
  "response": "AI応答"
}
```

### GET /health/

サーバー状態確認エンドポイント

## 🤖 AI モデル

現在 SKT KoGPT2 モデルを使用しており、`model_selector.py`で様々なモデルを選択できます：

- **SKT KoGPT2**: 韓国語に特化した GPT2 モデル
- **Microsoft Phi-4**: 軽量化されたインストラクトモデル

## 🎨 主要機能

- **韓国語サポート**: 韓国語に最適化された AI モデル
- **レスポンシブ UI**: モバイル・デスクトップ対応
- **ダークモード**: ユーザー設定によるテーマ切り替え
- **会話履歴**: 前の会話コンテキストの保持

## 🔧 開発ガイド

### 新しいモデルの追加

`api/model_selector.py`で新しいモデルを追加できます：

```python
NEW_MODEL = "your/model/path"

def select_model(self):
    if self.model_name == NEW_MODEL:
        # モデル設定
        pass
```

### フロントエンドのカスタマイズ

`front/app/page.tsx`で UI を修正できます。

## 📝 環境変数

### バックエンド

- `PYTHONUNBUFFERED=1`: ログの即座出力

### フロントエンド

- `NEXT_PUBLIC_API_URL`: バックエンド API URL (デフォルト: http://localhost:8000)

## 🐛 トラブルシューティング

### 一般的な問題

1. **ポート競合**

   - 8000 番ポートが使用中の場合: `docker-compose.yml`でポート変更
   - 3000 番ポートが使用中の場合: フロントエンドポート変更

2. **モデル読み込み失敗**

   - GPU メモリ不足: CPU モードに切り替え
   - モデルダウンロード失敗: インターネット接続確認

3. **ビルド失敗**
   - Docker キャッシュクリア: `docker-compose build --no-cache`

## 🤝 貢献

1. プロジェクトをフォーク
2. 機能ブランチを作成 (`git checkout -b feature/AmazingFeature`)
3. 変更をコミット (`git commit -m 'Add some AmazingFeature'`)
4. ブランチにプッシュ (`git push origin feature/AmazingFeature`)
5. プルリクエストを作成

## 📄 ライセンス

このプロジェクトは MIT ライセンスの下で配布されています。

## 🙏 謝辞

- [SKT](https://github.com/SKT-AI/KoGPT2) - 韓国語 GPT2 モデルの提供
- [Hugging Face](https://huggingface.co/) - トランスフォーマーライブラリ
- [FastAPI](https://fastapi.tiangolo.com/) - 高性能 Web フレームワーク
- [Next.js](https://nextjs.org/) - React フレームワーク

## 📞 連絡先

プロジェクトに関する質問や提案がある場合は、イシューを作成してください。

## 🎓 モデル 学習

### 学習 データ 準備

- カカオトウログファイルを `api/kakaotalkLog/` フォルダに配置
- または他のダイアログデータ 準備

### 学習 実行

```bash
cd api
python trainer.py
```

### 学習 オプション

- モデル 選択: 1. SKT KoGPT2, 2. Microsoft Phi-4
- 学習 パラメータ: エポック 5, バッチサイズ 自動設定
- チェックポイント: 500 ステップごとに保存

### 学習済みモデル 使用

- 学習 完了後 `model_selector.py` でパス設定
- API サーバー 再起動 必要

---

# AI Chat 애플리케이션

개인화된 AI 챗봇을 위한 풀스택 웹 애플리케이션입니다. SKT KoGPT2 모델을 기반으로 한 한국어 대화형 AI 시스템입니다.

## 🚀 프로젝트 개요

이 프로젝트는 카카오톡 대화 기록을 학습 데이터로 사용하여 사용자의 말투와 성격을 모방하는 개인화된 AI 챗봇을 만드는 것을 목적으로 합니다. FastAPI 백엔드와 Next.js 프론트엔드로 구성된 현대적인 웹 애플리케이션입니다.

### 📹 데모 영상

프로젝트의 동작을 확인하려면 다음 데모 영상을 시청해주세요:

![AI Chat Demo](sample.mp4)

_데모 영상: AI Chat 애플리케이션의 기본 기능과 사용 방법_

## 📁 프로젝트 구조

```
ai-chat/
├── api/                    # FastAPI 백엔드
│   ├── api.py             # 메인 API 서버
│   ├── model_selector.py  # AI 모델 선택 및 설정
│   ├── trainer.py         # 모델 훈련 로직
│   ├── requirements.txt   # Python 의존성
│   └── Dockerfile         # 백엔드 컨테이너 설정
├── front/                 # Next.js 프론트엔드
│   ├── app/              # Next.js 앱 디렉토리
│   ├── package.json      # Node.js 의존성
│   └── Dockerfile        # 프론트엔드 컨테이너 설정
├── docker-compose.yml    # 전체 시스템 오케스트레이션
└── results/              # 훈련된 모델 체크포인트
```

## 🛠️ 기술 스택

### 백엔드

- **FastAPI**: 고성능 Python 웹 프레임워크
- **PyTorch**: 딥러닝 프레임워크
- **Transformers**: Hugging Face 트랜스포머 라이브러리
- **SKT KoGPT2**: 한국어 GPT2 모델

### 프론트엔드

- **Next.js 15**: React 기반 풀스택 프레임워크
- **TypeScript**: 타입 안전성
- **Tailwind CSS**: 유틸리티 우선 CSS 프레임워크

### 인프라

- **Docker**: 컨테이너화
- **Docker Compose**: 멀티 컨테이너 오케스트레이션

## 🚀 시작하기

### 사전 요구사항

- Docker 및 Docker Compose
- Node.js 18+ (로컬 개발용)
- Python 3.8+ (로컬 개발용)

### Docker를 사용한 실행 (권장)

1. 저장소 클론

```bash
git clone <repository-url>
cd ai-chat
```

2. Docker Compose로 실행

```bash
docker-compose up --build
```

3. 브라우저에서 접속

- 프론트엔드: http://localhost:3000
- 백엔드 API: http://localhost:8000

### 로컬 개발 환경

#### 백엔드 설정

```bash
cd api
pip install -r requirements.txt
python api.py
```

#### 프론트엔드 설정

```bash
cd front
npm install
npm run dev
```

## 📡 API 엔드포인트

### POST /chat/

챗봇과 대화를 위한 메인 엔드포인트

**요청 본문:**

```json
{
  "prompt": "사용자 메시지",
  "previous_message": "이전 대화 내용",
  "max_length": 300
}
```

**응답:**

```json
{
  "response": "AI 응답"
}
```

### GET /health/

서버 상태 확인 엔드포인트

## 🤖 AI 모델

현재 SKT KoGPT2 모델을 사용하고 있으며, `model_selector.py`에서 다양한 모델을 선택할 수 있습니다:

- **SKT KoGPT2**: 한국어에 특화된 GPT2 모델
- **Microsoft Phi-4**: 경량화된 인스트럭트 모델

## 🎨 주요 기능

- **한국어 지원**: 한국어에 최적화된 AI 모델
- **반응형 UI**: 모바일 및 데스크톱 지원
- **다크 모드**: 사용자 선호도에 따른 테마 전환
- **대화 기록**: 이전 대화 컨텍스트 유지

## 🔧 개발 가이드

### 새로운 모델 추가

`api/model_selector.py`에서 새로운 모델을 추가할 수 있습니다:

```python
NEW_MODEL = "your/model/path"

def select_model(self):
    if self.model_name == NEW_MODEL:
        # 모델 설정
        pass
```

### 프론트엔드 커스터마이징

`front/app/page.tsx`에서 UI를 수정할 수 있습니다.

## 📝 환경 변수

### 백엔드

- `PYTHONUNBUFFERED=1`: 로그 즉시 출력

### 프론트엔드

- `NEXT_PUBLIC_API_URL`: 백엔드 API URL (기본값: http://localhost:8000)

## 🐛 문제 해결

### 일반적인 문제들

1. **포트 충돌**

   - 8000번 포트가 사용 중인 경우: `docker-compose.yml`에서 포트 변경
   - 3000번 포트가 사용 중인 경우: 프론트엔드 포트 변경

2. **모델 로딩 실패**

   - GPU 메모리 부족: CPU 모드로 전환
   - 모델 다운로드 실패: 인터넷 연결 확인

3. **빌드 실패**
   - Docker 캐시 클리어: `docker-compose build --no-cache`

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 🙏 감사의 말

- [SKT](https://github.com/SKT-AI/KoGPT2) - 한국어 GPT2 모델 제공
- [Hugging Face](https://huggingface.co/) - 트랜스포머 라이브러리
- [FastAPI](https://fastapi.tiangolo.com/) - 고성능 웹 프레임워크
- [Next.js](https://nextjs.org/) - React 프레임워크

## 📞 연락처

프로젝트에 대한 질문이나 제안사항이 있으시면 이슈를 생성해주세요.

## 🎓 모델 학습

### 학습 데이터 준비

- 카카오톡 로그 파일을 `api/kakaotalkLog/` 폴더에 배치
- 또는 다른 대화 데이터 준비

### 학습 실행

```bash
cd api
python trainer.py
```

### 학습 옵션

- 모델 선택: 1. SKT KoGPT2, 2. Microsoft Phi-4
- 학습 파라미터: 에포크 5, 배치 크기 자동 설정
- 체크포인트: 500 스텝마다 저장

### 학습된 모델 사용

- 학습 완료 후 `model_selector.py`에서 경로 설정
- API 서버 재시작 필요
