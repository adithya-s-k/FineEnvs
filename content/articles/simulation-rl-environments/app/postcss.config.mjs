// PostCSS config enabling Custom Media Queries
// Allows usage of: @media (--bp-content-collapse) { ... }

import postcssCustomMedia from 'postcss-custom-media';
import postcssPresetEnv from 'postcss-preset-env';

export default {
  plugins: [
    postcssCustomMedia(),
    postcssPresetEnv({
      stage: 0
    })
  ]
};
