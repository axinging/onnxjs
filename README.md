npx cross-env ELECTRON_GET_USE_PROXY=true GLOBAL_AGENT_HTTPS_PROXY=http://proxy:90 npm install electron --legacy-peer-deps
npx cross-env ELECTRON_GET_USE_PROXY=true GLOBAL_AGENT_HTTPS_PROXY=http://proxy:90 npm install got --legacy-peer-deps
npx cross-env ELECTRON_GET_USE_PROXY=true GLOBAL_AGENT_HTTPS_PROXY=http://proxy:90 npm install @electron/get  --legacy-peer-deps
npm install ts-loader --legacy-peer-deps
npm audit fix --force
npm install --legacy-peer-deps
npm ci --legacy-peer-deps
npm run build:bundle

