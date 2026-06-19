# Build Vite app, serve via nginx with /api proxy to backend.
FROM node:20-alpine AS build

WORKDIR /app
COPY package.json ./
RUN npm install

COPY index.html vite.config.js ./
COPY public/ ./public/
COPY src/ ./src/

# /api is proxied by nginx to the backend container (see nginx.conf)
ARG VITE_API_URL=/api
ENV VITE_API_URL=$VITE_API_URL

RUN npm run build

FROM nginx:1.27-alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
