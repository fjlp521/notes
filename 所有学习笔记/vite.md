<h1 style="text-align:center;">vite</h1>

#### 一、vite 工作原理

###### 1、开发环境

- 重点：**开发环境下**无需打包，无需打包，无需打包
- webpack 工作流程：先集合所有文件，然后打包编译，最后搭建服务器；而 vite 工作流程：先搭建服务器，然后访问哪个文件就加载哪个文件（建立在以下基础上：浏览器开始原生支持 ES 模块），比如访问 Home 组件，就会加载 Home.vue，但是此次响应会设置响应头：`Content-Type: application/javascript`
- 当修改某个文件时，webpack 会重新打包，而 vite 只是监测到某个文件发生了变化，而不会执行任何操作，所以在开发环境下 vite 启动速度很快

###### 2、生产环境

- 与 webpack 类似，也需要打包，两者差不多

#### 二、vite 迁移指南

###### 1、vue3-cli 项目迁移到 vue3-vite

1. 新建一个空的 vue3-vite 项目
2. 使用原项目 package.json 中的 "dependencies" 覆盖掉新 vue3-vite 项目 package.json 中的 "dependencies" 
3. `npm install`
4.  使用原项目中的 src 文件夹覆盖掉新 vue3-vite 项目中的 src 文件夹

注意：vite 中不支持使用 "@" 代替 "src"，因此在模块引入方面可能需要做一些修改

###### 2、vue2-cli 项目迁移到 vue2-vite

1. 新建一个空的 vue3-vite 项目

2. 使用原项目 package.json 中的 "dependencies" 覆盖掉新 vue3-vite 项目 package.json 中的 "dependencies" 

3. `npm install`

4.  使用原项目中的 src 文件夹覆盖掉新 vue3-vite 项目中的 src 文件夹

5. 将新 vue3-vite 项目中配置的 vue3 插件更换为 vue2 插件（在 vite.config.js 中修改）

   ```js
   //修改前的 vite.config.js
   import { defineConfig } from 'vite'
   import vue from '@vitejs/plugin-vue'	//vue3插件
   
   export default defineConfig({
     plugins: [vue()],
   })
   
   //修改后的 vite.config.js
   import { defineConfig } from 'vite'
   import { createVuePlugin } from 'vite-plugin-vue2'	//vue2插件
   
   export default defineConfig({
     plugins: [createVuePlugin(/* options */)],
   })
   ```

注意：如果想在 react 项目中应用 vite ，就引入 react 插件

