<h1 style="text-align:center;">HTML</h1>

##### 1、checkbox 的 checked 和 value

- checkbox 是否选中与 checked 属性有关，checked 为 true 时选中，为 false 时未选中，与 value 无关
- value 可以理解为整个 checkbox 的名字，默认值为 on
- 若表单提交时，checkbox 未勾选，则提交的值并非为 value=unchecked；此时的值不会被提交到服务器
- 对于纯HTML，<input *type*="checkbox" *checked*='xxx'>，只要标签内写了checked，不管xxx值是什么，该复选框初始时都会处于选中状态，当然之后将checked设置为false即可取消勾选
- 对于Vue，<input *type*="checkbox"  :*checked*='xxx'>，会根据xxx的值设置checked，使其选中或未选中

##### 2、H5新增标签

- 语义化标签
  - header、nav、main、article、section、aside、footer
  - 这种语义化标准主要针对搜索引擎
  - 新标签在页面中可以使用多次
  - 移动端更适合使用这些标签

- 音频

  - audio、video

- input: xxx

- datalist

  ```html
  <input type="text" list="test">
  <datalist id="test">
      <option value="广州">gz</option>
      <option value="北京">bj</option>
  </datalist>
  ```

- 表单标签的一些新增属性：required、autofocus