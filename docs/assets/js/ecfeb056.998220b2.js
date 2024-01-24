"use strict";(self.webpackChunkscanmate=self.webpackChunkscanmate||[]).push([[497],{3905:(e,t,a)=>{a.d(t,{Zo:()=>l,kt:()=>d});var n=a(7294);function i(e,t,a){return t in e?Object.defineProperty(e,t,{value:a,enumerable:!0,configurable:!0,writable:!0}):e[t]=a,e}function r(e,t){var a=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),a.push.apply(a,n)}return a}function s(e){for(var t=1;t<arguments.length;t++){var a=null!=arguments[t]?arguments[t]:{};t%2?r(Object(a),!0).forEach((function(t){i(e,t,a[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(a)):r(Object(a)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(a,t))}))}return e}function o(e,t){if(null==e)return{};var a,n,i=function(e,t){if(null==e)return{};var a,n,i={},r=Object.keys(e);for(n=0;n<r.length;n++)a=r[n],t.indexOf(a)>=0||(i[a]=e[a]);return i}(e,t);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);for(n=0;n<r.length;n++)a=r[n],t.indexOf(a)>=0||Object.prototype.propertyIsEnumerable.call(e,a)&&(i[a]=e[a])}return i}var m=n.createContext({}),c=function(e){var t=n.useContext(m),a=t;return e&&(a="function"==typeof e?e(t):s(s({},t),e)),a},l=function(e){var t=c(e.components);return n.createElement(m.Provider,{value:t},e.children)},g="mdxType",u={inlineCode:"code",wrapper:function(e){var t=e.children;return n.createElement(n.Fragment,{},t)}},p=n.forwardRef((function(e,t){var a=e.components,i=e.mdxType,r=e.originalType,m=e.parentName,l=o(e,["components","mdxType","originalType","parentName"]),g=c(a),p=i,d=g["".concat(m,".").concat(p)]||g[p]||u[p]||r;return a?n.createElement(d,s(s({ref:t},l),{},{components:a})):n.createElement(d,s({ref:t},l))}));function d(e,t){var a=arguments,i=t&&t.mdxType;if("string"==typeof e||i){var r=a.length,s=new Array(r);s[0]=p;var o={};for(var m in t)hasOwnProperty.call(t,m)&&(o[m]=t[m]);o.originalType=e,o[g]="string"==typeof e?e:i,s[1]=o;for(var c=2;c<r;c++)s[c]=a[c];return n.createElement.apply(null,s)}return n.createElement.apply(null,a)}p.displayName="MDXCreateElement"},9422:(e,t,a)=>{a.r(t),a.d(t,{assets:()=>m,contentTitle:()=>s,default:()=>u,frontMatter:()=>r,metadata:()=>o,toc:()=>c});var n=a(7462),i=(a(7294),a(3905));const r={sidebar_position:4,id:"Image Matching",description:"Match similar images.",slug:"/under-the-hood/image-matching"},s=void 0,o={unversionedId:"under-the-hood/Image Matching",id:"under-the-hood/Image Matching",title:"Image Matching",description:"Match similar images.",source:"@site/docs/under-the-hood/image-matching.md",sourceDirName:"under-the-hood",slug:"/under-the-hood/image-matching",permalink:"/under-the-hood/image-matching",draft:!1,editUrl:"https://github.com/ZiadMansourM/photogrammetry/tree/main/docs/under-the-hood/image-matching.md",tags:[],version:"current",sidebarPosition:4,frontMatter:{sidebar_position:4,id:"Image Matching",description:"Match similar images.",slug:"/under-the-hood/image-matching"},sidebar:"tutorialSidebar",previous:{title:"Compute SIFT",permalink:"/under-the-hood/compute-sift"},next:{title:"Feature Matching",permalink:"/under-the-hood/feature-matching"}},m={},c=[{value:"\ud83d\udcdd Image Matching/Stitching",id:"-image-matchingstitching",level:2}],l={toc:c},g="wrapper";function u(e){let{components:t,...a}=e;return(0,i.kt)(g,(0,n.Z)({},l,a,{components:t,mdxType:"MDXLayout"}),(0,i.kt)("h2",{id:"-image-matchingstitching"},"\ud83d\udcdd Image Matching/Stitching"),(0,i.kt)("ul",{className:"contains-task-list"},(0,i.kt)("li",{parentName:"ul",className:"task-list-item"},(0,i.kt)("input",{parentName:"li",type:"checkbox",checked:!0,disabled:!0})," ","Match similar images together into a cluster."),(0,i.kt)("li",{parentName:"ul",className:"task-list-item"},(0,i.kt)("input",{parentName:"li",type:"checkbox",checked:!0,disabled:!0})," ","Images in a cluster has to see the very same points.")),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-py"},"@timeit\ndef image_matching(images_obj: Images, overwrite:bool =False, **kwargs) -> None:\n    def load_image(image_path, target_size=(224, 224)):\n        img = keras_image.load_img(image_path, target_size=target_size)\n        img = keras_image.img_to_array(img)\n        img = np.expand_dims(img, axis=0)\n        img = preprocess_input(img)\n        return img\n\n    image_set_name = kwargs['image_set_name']\n    image_dir = f'../data/{image_set_name}/images'\n    image_files = os.listdir(image_dir)\n    images = [load_image(os.path.join(image_dir, f)) for f in image_files]\n    images = np.vstack(images)\n\n    ssl_context = ssl.create_default_context(cafile=certifi.where())\n    ssl._create_default_https_context = ssl._create_unverified_context\n    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')\n    features = model.predict(images)\n\n    kmeans = KMeans(n_clusters=images_obj.num_clusters, random_state=42)\n    clusters = kmeans.fit_predict(features)\n\n    for i, cluster in enumerate(clusters):\n        if cluster not in images_obj.similar_images:\n            images_obj.similar_images[cluster] = []\n        images_obj.similar_images[cluster].append(images_obj[int(image_files[i].split(\".\")[0])])\n")))}u.isMDXComponent=!0}}]);